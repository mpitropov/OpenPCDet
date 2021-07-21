import copy
import torch
import numpy as np
from collections import defaultdict
from numpy.core.fromnumeric import repeat
from random import random

from ...utils import box_utils, common_utils
from .kitti_dataset_var import KittiDatasetVAR
from ..dataset import DatasetTemplate
from ..processor.data_processor import DataProcessor

class KittiDatasetMIMOVARB(DatasetTemplate):
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

        self.NUM_HEADS = dataset_cfg.NUM_HEADS
        self.MAX_POINTS_PER_VOXEL = dataset_cfg.DATA_PROCESSOR[2]['MAX_POINTS_PER_VOXEL']
        self.INPUT_REPETITION = dataset_cfg.INPUT_REPETITION
        self.BATCH_REPETITION = dataset_cfg.BATCH_REPETITION

        self.rng = np.random.default_rng()

        self.kitti_dataset = KittiDatasetVAR(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            root_path=root_path,
            training=training,
            logger=logger,
        )

        # Create two different dataset proccesors
        # First one has performs masking and second will shuffle pts and voxelize
        data_processor_cfg_masking = \
            [x for x in self.dataset_cfg.DATA_PROCESSOR if x.NAME == 'mask_points_and_boxes_outside_range']
        data_processor_cfg_shffl_voxelize = \
            [x for x in self.dataset_cfg.DATA_PROCESSOR if x.NAME != 'mask_points_and_boxes_outside_range']
        self.data_processor_masking = DataProcessor(
            data_processor_cfg_masking, point_cloud_range=self.point_cloud_range, training=self.training
        )
        self.data_processor_shffl_voxelize = DataProcessor(
            data_processor_cfg_shffl_voxelize, point_cloud_range=self.point_cloud_range, training=self.training
        )

    def __len__(self):
        return len(self.kitti_dataset)

    def __getitem__(self, index):
        if self.kitti_dataset._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_dataset.kitti_infos)

        info = copy.deepcopy(self.kitti_dataset.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']

        points = self.kitti_dataset.get_lidar(sample_idx)
        calib = self.kitti_dataset.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.kitti_dataset.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.kitti_dataset.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            road_plane = self.kitti_dataset.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        # Custom prepare data function
        # Which masks points but only performs shuffle & voxelize during testing
        data_dict = self.prepare_data(data_dict=input_dict,
                                        head_dataset=self.kitti_dataset)
        data_dict['image_shape'] = img_shape
        return data_dict

    def prepare_data(self, data_dict, head_dataset):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if head_dataset.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in head_dataset.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = head_dataset.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], head_dataset.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([head_dataset.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = head_dataset.point_feature_encoder.forward(data_dict)

        data_dict = self.self.data_processor_masking.forward(
            data_dict=data_dict
        )

        if not self.training:
            data_dict = self.data_processor_shffl_voxelize.forward(
                data_dict=data_dict
            )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(head_dataset.__len__())
            return head_dataset.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    # During training the pointclouds must be combined and voxelized together
    def modify_batch(self, batch_list):
        data_dict = defaultdict(list)
        batch_size = len(batch_list)
        batch_repetitions = 1
        if self.training:
            batch_repetitions = self.BATCH_REPETITION
        main_shuffle = np.tile( np.arange(batch_size), batch_repetitions)
        self.rng.shuffle(main_shuffle)
        to_shuffle = int(len(main_shuffle) * (1. - self.INPUT_REPETITION) )

        # Each row contains a different grouping of frames
        # Each column is a different detection head
        frame_list = []
        for i in range(self.NUM_HEADS):
            rand_portion = copy.deepcopy(main_shuffle[:to_shuffle])
            self.rng.shuffle(rand_portion)
            frame_list.append(np.concatenate([rand_portion, main_shuffle[to_shuffle:]]))
        frame_list = np.transpose(frame_list)

        for frame_group_index in range(len(frame_list)):
            # Loop through frames in this grouping
            for head_id in range(self.NUM_HEADS):
                batch_list_index = frame_list[frame_group_index][head_id]
                # Add non voxel components
                for key, val in batch_list[batch_list_index].items():
                    if key in ['voxels', 'voxel_coords', 'voxel_num_points']:
                        continue
                    data_dict[key].append(val)

            # Create tmp dict with points from all frames within the group
            point_cloud_list = []
            for head_id in range(self.NUM_HEADS):
                batch_list_index = frame_list[frame_group_index][head_id]
                point_cloud_list.append(batch_list[batch_list_index]['points'])
            # Create points var with all points
            points = np.concatenate(point_cloud_list)
            tmp_dict = { 'points': points }

            # Shuffle and voxelize the pointcloud
            tmp_dict = self.data_processor_shffl_voxelize.forward(
                data_dict=tmp_dict
            )

            # Add the voxel stuff to the data_dict
            data_dict['voxels'].append(tmp_dict['voxels'])
            data_dict['voxel_coords'].append(tmp_dict['voxel_coords'])
            data_dict['voxel_num_points'].append(tmp_dict['voxel_num_points'])

        return data_dict

    # This collate_batch function is modified from the one in dataset.py
    # Instead of receiving a list of data_dicts (N),
    # it receives a list of lists of data_dicts (N, number of heads)
    def collate_batch(self, batch_list, _unused=False):
        batch_repetitions = 1
        if self.training:
            batch_repetitions = self.BATCH_REPETITION

        if self.training:
            data_dict = self.modify_batch(batch_list)
            # N * number of heads * batch repetition
            batch_size = len(batch_list) * self.NUM_HEADS * batch_repetitions
        else:
            data_dict = defaultdict(list)
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    # Voxel stuff is added once
                    # Other stuff must be added for each head
                    if key in ['voxels', 'voxel_coords', 'voxel_num_points']:
                        data_dict[key].append(val)
                    else:
                        for head_id in range(self.NUM_HEADS):
                            data_dict[key].append(val)
            # N * number of heads * batch repetition
            batch_size = len(batch_list) * self.NUM_HEADS * batch_repetitions

        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size

        return ret

    # We can use the method from one of our heads
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
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
        FRAME_NUM = 0 # Must have eval set to batch size of 1
        ret_dict = self.kitti_dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path)

        # Generate prediction dicts for each head
        ret_dict_list = []
        for i in range(self.NUM_HEADS):
            ret_dict_list.append(self.kitti_dataset.generate_prediction_dicts( \
                batch_dict, pred_dicts[FRAME_NUM]['pred_dicts_list'][i], class_names, output_path)[FRAME_NUM])
        ret_dict[FRAME_NUM]['post_nms_head_outputs'] = ret_dict_list

        return ret_dict

    # Must also use this method from one of our heads
    def evaluation(self, det_annos, class_names, **kwargs):
        return self.kitti_dataset.evaluation(det_annos, class_names, **kwargs)
