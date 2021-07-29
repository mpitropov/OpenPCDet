import os
import sys
import pickle
import copy
import numpy as np
import json
import math
from skimage import io
from pathlib import Path
import torch

from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.config import cfg
from pcdet.datasets.cadc import cadc_calibration

import time

class CadcDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.imagesets_path = Path(self.dataset_cfg.DATA_PATH)
        if root_path != None: # Image sets information
            self.imagesets_path = root_path
        self.root_path = Path(self.dataset_cfg.DATA_PATH) # Path to the actual data
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.imagesets_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip().split(' ') for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.cadc_infos = []
        self.include_cadc_data(self.mode)

    def include_cadc_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading CADC dataset')
        cadc_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                cadc_infos.extend(infos)

        self.cadc_infos.extend(cadc_infos)

        if self.logger is not None:
            self.logger.info('Total samples for CADC dataset: %d' % (len(cadc_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.imagesets_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip().split(' ') for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, sample_idx):
        date, set_num, idx = sample_idx
        lidar_file = os.path.join(self.root_path, date, set_num, 'labeled', 'lidar_points', 'data', '%s.bin' % idx)
        assert os.path.exists(lidar_file)
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return points

    def get_image_shape(self, sample_idx):
        if len(sample_idx) != 3:
            print("get_image_shape wrong len")
            print(sample_idx)
            print(len(sample_idx))
            exit
        date, set_num, idx = sample_idx
        img_file = os.path.join(self.root_path, date, set_num, 'labeled', 'image_00', 'data', '%s.png' % idx)
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, sample_idx):
        date, set_num, idx = sample_idx
        label_file = os.path.join(self.root_path, date, set_num, '3d_ann.json')
        assert os.path.exists(label_file)
        return json.load(open(label_file, 'r'))

    def get_calib(self, sample_idx):
        date, set_num, idx = sample_idx
        calib_path = os.path.join(self.root_path, date, 'calib')
        assert os.path.exists(calib_path)
        return cadc_calibration.Calibration(calib_path)

    def get_road_plane(self, idx):
        """
        plane_file = os.path.join(self.root_path, 'planes', '%s.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane
        """
        # Currently unsupported in CADC
        raise NotImplementedError

    # Get point_count, distance, score threshold from yaml file
    def get_threshold(self):
        params = self.dataset_cfg.FILTER_CRITERIA['filter_by_min_points']
        point_count_threshold = {}
        for elem in params:
            key, val = elem.split(':')
            point_count_threshold[key] = int(val)

        distance_threshold = self.dataset_cfg.FILTER_CRITERIA['distance']
        score_threshold = self.dataset_cfg.FILTER_CRITERIA['score']
        return point_count_threshold, distance_threshold, score_threshold

    def get_annotation_from_label(self, calib, sample_idx):
        date, set_num, idx = sample_idx
        obj_list_temp = self.get_label(sample_idx)[int(idx)]['cuboids']
        point_count_threshold, distance_threshold, _ = self.get_threshold()

        obj_list = []
        closest_ratio_to_criteria, closest_idx_to_criteria = 0.0, 0

        # Filter out objects
        for idx, obj in enumerate(obj_list_temp):
            # Filter by label
            if (obj['label'] not in ['Car', 'Pedestrian', 'Truck']):
                continue
            # Set Pickup_Truck label as top label
            if obj['label'] == 'Truck':
                if obj['attributes']['truck_type'] == 'Pickup_Truck':
                    obj['label'] = 'Pickup_Truck'
                else:
                    continue
            # filter by point_count
            if obj['points_count'] == 0:
                ratio_to_criteria = obj['points_count'] / point_count_threshold[obj['label']]
                if ratio_to_criteria > closest_ratio_to_criteria:
                    closest_idx_to_criteria = idx
                continue
            # filter by distance
            x, y, z = obj['position']['x'],obj['position']['y'],obj['position']['z']
            if abs(x) > distance_threshold or abs(y) > distance_threshold:
                continue
            obj_list.append(obj)
        
        # If there is no object in the frame satisfying the filter criteria, add
        # the object that is the closest to the filter criteria to avoid empty
        # annotations error downstream in other functions.
        if len(obj_list) == 0:
            obj_list.append(obj_list_temp[closest_idx_to_criteria])

        annotations = {}
        annotations['name'] = np.array([obj['label'] for obj in obj_list])
        annotations['num_points_in_gt'] = np.array([obj['points_count'] for obj in obj_list])
        
        loc_lidar = np.array([[obj['position']['x'],obj['position']['y'],obj['position']['z']] for obj in obj_list])
        # Scale AI gives x as width and y as length, we must switch these
        dims = np.array([[obj['dimensions']['y'],obj['dimensions']['x'],obj['dimensions']['z']] for obj in obj_list])
        rots = np.array([obj['yaw'] for obj in obj_list])
        gt_boxes_lidar = np.concatenate([loc_lidar, dims, rots[..., np.newaxis]], axis=1)
        annotations['gt_boxes_lidar'] = gt_boxes_lidar
        
        # in camera 0 frame. Probably meaningless as most objects aren't in frame.
        annotations['location'] = calib.lidar_to_rect(loc_lidar)
        annotations['rotation_y'] = rots
        # Scale AI gives x as width and y as length, we must switch these
        annotations['dimensions'] = np.array([[obj['dimensions']['y'], obj['dimensions']['z'], obj['dimensions']['x']] for obj in obj_list])  # lhw format

        # Should not be modifying the original lidar positions
        gt_boxes_copy = copy.deepcopy(gt_boxes_lidar)
        gt_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes_copy, calib)
        gt_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            gt_boxes_camera, calib, image_shape=self.get_image_shape(sample_idx)
        )

        # Calculate occluded levels for testing
        point_levels = [15, 5, 1]
        dist_levels = [30.0, 50.0, 75.0] # 75.0 will cover max diagonal distance
        occluded_list = []
        for obj in obj_list:
            dist_to_obj = math.sqrt(obj['position']['x']**2 + obj['position']['y']**2)
            if obj['points_count'] >= point_levels[0] and dist_to_obj <= dist_levels[0]:
                occluded_list.append(0) # fully visible
            elif obj['points_count'] >= point_levels[1] and dist_to_obj <= dist_levels[1]:
                occluded_list.append(1) # partly occluded
            elif obj['points_count'] >= point_levels[2] and dist_to_obj <= dist_levels[2]:
                occluded_list.append(2) # largely occluded
            else:
                occluded_list.append(3) # unknown

        annotations['occluded'] = np.array(occluded_list)

        # Currently unused for CADC, and don't make too much since as we primarily use 360 degree 3d LIDAR boxes.
        annotations['score'] = np.array([1 for _ in obj_list])
        annotations['difficulty'] = np.array([0 for obj in obj_list], np.int32)
        annotations['truncated'] = np.array([0 for _ in obj_list])
        annotations['alpha'] = np.array([-np.arctan2(-gt_boxes_lidar[i][1], gt_boxes_lidar[i][0]) + gt_boxes_camera[i][6] for i in range(len(obj_list))]) 
        annotations['bbox'] = gt_boxes_img
        
        return annotations
    
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        '''
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param img_shape:
        :return:
        '''
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            
            print('%s sample_idx: %s ' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)
            
            calib_info = {'T_IMG_CAM0': calib.t_img_cam[0], 'T_CAM_LIDAR': calib.t_cam_lidar[0]}

            info['calib'] = calib_info

            if has_label:
                annotations = self.get_annotation_from_label(calib, sample_idx)
                info['annos'] = annotations
            return info

        # temp = process_single_scene(self.sample_id_list[0])
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('cadc_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes[:,:7])
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%s_%s_%d.bin' % (sample_idx[0], sample_idx[1], sample_idx[2], names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        # Return a list of index which satisfies score and distance criteria
        def filter_criteria(pred_scores, pred_locations, score_threshold, distance_threshold):
            x, y, z = pred_locations[:,0], pred_locations[:,1], pred_locations[:,2]
            distance = np.sqrt(np.square(x)+np.square(y)+np.square(z))
            index = []
            for i in range(pred_scores.shape[0]):
                if pred_scores[i] >= score_threshold and distance[i] < distance_threshold:
                    index.append(True)
                else:
                    index.append(False)
            index = np.array(index)
            return index

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            # Should not be modifying the original lidar positions
            pred_boxes_copy = copy.deepcopy(pred_boxes)
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes_copy, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['sample_idx'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        assert 'annos' in self.cadc_infos[0].keys()
        import pcdet.datasets.kitti.kitti_object_eval_python.eval as kitti_eval

        if 'annos' not in self.cadc_infos[0]:
            return 'None', {}

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.cadc_infos]

        for i in range(len(eval_gt_annos)):
            boxes3d_lidar = np.array(eval_gt_annos[i]['gt_boxes_lidar'])
            # Original get_official_eval_result is for kitti or nuscenes depending on the setting
            # Convert cadc lidar data to in the same axes as the nuscenes
            boxes3d_lidar = boxes3d_lidar[:,[1,0,2,3,4,5,6]]
            boxes3d_lidar[:,0] *= -1
            eval_gt_annos[i]['location'] = boxes3d_lidar[:,:3]
            eval_gt_annos[i]['dimensions'] = boxes3d_lidar[:,3:6]
            eval_gt_annos[i]['rotation_y'] = boxes3d_lidar[:,6]
            
            # Arbituarily set bbox as it's not applicable in cadc
            # Note you need to make sure the height of the box > 40 or it will be filtered out
            bbox = []
            for j in range(len(eval_gt_annos[i]['bbox'])):
                bbox.append(np.array([0,0,50,50]))
            eval_gt_annos[i]['bbox'] = bbox

        for i in range(len(eval_det_annos)):
            boxes3d_lidar = np.array(eval_det_annos[i]['boxes_lidar'])
            boxes3d_lidar = boxes3d_lidar[:,[1,0,2,3,4,5,6]]
            boxes3d_lidar[:,0] *= -1
            eval_det_annos[i]['location'] = boxes3d_lidar[:,:3]
            eval_det_annos[i]['dimensions'] = boxes3d_lidar[:,3:6]
            eval_det_annos[i]['rotation_y'] = boxes3d_lidar[:,6]
            bbox = []
            for j in range(len(eval_det_annos[i]['bbox'])):
                bbox.append(np.array([0,0,50,50]))
            eval_det_annos[i]['bbox'] = np.array(bbox)

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names, 
            z_axis=2, z_center=0.5)

        return ap_result_str, ap_dict

    def __len__(self):
        return len(self.cadc_infos)

    def __getitem__(self, index):
        self.start_time = time.time()
        info = copy.deepcopy(self.cadc_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if cfg.DATA_CONFIG.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'sample_idx': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']

            # Create mask to filter annotations during training
            if self.training and self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (annos['num_points_in_gt'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            gt_names = annos['name'] if mask is None else annos['name'][mask]
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar'] if mask is None else annos['gt_boxes_lidar'][mask]
            else:
                # This should not run, although the code should look somewhat like this
                raise NotImplementedError
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict 

def create_cadc_infos(dataset_cfg, class_names, data_path, save_path, workers=8):
    dataset = CadcDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('cadc_infos_%s.pkl' % train_split)
    val_filename = save_path / ('cadc_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'cadc_infos_trainval.pkl'
    test_filename = save_path / 'cadc_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    cadc_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(cadc_infos_train, f)
    print('Cadc info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    cadc_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(cadc_infos_val, f)
    print('Cadc info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(cadc_infos_train + cadc_infos_val, f)
    print('Cadc info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    cadc_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(cadc_infos_test, f)
    print('Cadc info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_cadc_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_cadc_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Pickup_Truck'],
            data_path=ROOT_DIR / 'data' / 'cadc',
            save_path=Path(dataset_cfg.DATA_PATH)
        )
