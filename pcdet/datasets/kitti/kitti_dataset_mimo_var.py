import copy
import numpy as np
from collections import defaultdict
from numpy.core.fromnumeric import repeat
from random import random

from ...utils import box_utils, common_utils
from .kitti_dataset_var import KittiDatasetVAR
from ..dataset import DatasetTemplate
from ..processor.data_processor import DataProcessor

class KittiDatasetMIMOVAR(DatasetTemplate):
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

        self.NUM_HEADS = dataset_cfg.NUM_HEADS
        self.MAX_POINTS_PER_VOXEL = dataset_cfg.DATA_PROCESSOR[2]['MAX_POINTS_PER_VOXEL']
        self.INPUT_REPETITION = dataset_cfg.INPUT_REPETITION
        self.BATCH_REPETITION = dataset_cfg.BATCH_REPETITION

        # Init functions
        self.current_epoch = 0
        self.prev_index = 0

        # Initiate dataset for each head
        self.head_datasets = []
        self.head_seed_offsets = []
        for i in range(0, self.NUM_HEADS):
            # TODO Generalize __all__[dataset_cfg.DATASET]
            self.head_datasets.append(
                KittiDatasetVAR(
                    dataset_cfg=dataset_cfg,
                    class_names=class_names,
                    root_path=root_path,
                    training=training,
                    logger=logger,
                )
            )
            # Each head has offset, ex. 3 heads [0, 100000, 200000]
            self.head_seed_offsets.append(self.current_epoch + i * 100000)

        # Initiate random array for each head
        self.initial_array = np.arange(0, len(self.head_datasets[0]), 1)
        self.rng_list = []
        self.head_random_index_arrs = []
        for i in range(0, self.NUM_HEADS):
            rng = np.random.default_rng(self.head_seed_offsets[i])
            self.head_random_index_arrs.append(rng.permutation(self.initial_array))
            self.rng_list.append(rng)

    def __len__(self):
        return len(self.head_datasets[0])

    def __getitem__(self, index):
        if self.prev_index > index:
            self.current_epoch += 1
            for i in range(0, self.NUM_HEADS):
                # Update seed and use it
                self.head_seed_offsets[i] = self.current_epoch + i * 100000
                rng = np.random.default_rng(self.head_seed_offsets[i])
                # Store new random array
                self.head_random_index_arrs[i] = rng.permutation(self.initial_array)
                self.rng_list[i] = rng

        self.prev_index = index
        # print("get index", index)

        data_dict_arr = [] # Store data dictionary for each head
        point_cloud_list = [] # Store point clouds to combine

        # If less than self.INPUT_REPETITION, sync head frames to head 0
        random_num = random()

        # For each head run the __getitem__ function in kitti_dataset.py
        for head_id in range(len(self.head_datasets)):
            head_dataset = self.head_datasets[head_id]

            if self.training:
                # When training we want to get the random index for this head
                if random_num < self.INPUT_REPETITION:
                    # however in this case, we sync all data to head 0
                    rand_index = self.head_random_index_arrs[0][index]
                else:
                    # All heads load different data
                    rand_index = self.head_random_index_arrs[head_id][index]
            else:
                # When testing we want all heads to get the same data
                rand_index = index

            if head_dataset._merge_all_iters_to_one_epoch:
                print("Not tested")
                rand_index = rand_index % len(head_dataset.kitti_infos)
                raise NotImplementedError

            info = copy.deepcopy(head_dataset.kitti_infos[rand_index])
            sample_idx = info['point_cloud']['lidar_idx']

            points = head_dataset.get_lidar(sample_idx)
            calib = head_dataset.get_calib(sample_idx)

            img_shape = info['image']['image_shape']
            if head_dataset.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = head_dataset.get_fov_flag(pts_rect, img_shape, calib)
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
                road_plane = head_dataset.get_road_plane(sample_idx)
                if road_plane is not None:
                    input_dict['road_plane'] = road_plane

            # Custom prepare data function with two modifications
            # 1. Takes in head dataset to call instead of self. and head_id
            # 2. Add head id column to points
            # 3. Calls our own data processor which does not voxelize
            data_dict = self.prepare_data(data_dict=input_dict,
                                            head_dataset=head_dataset, head_id=head_id)
            data_dict['image_shape'] = img_shape

            data_dict_arr.append(data_dict)
            point_cloud_list.append(data_dict_arr[head_id]['points'])

            # If syncing all data then we only need one point cloud
            if not self.training or random_num < self.INPUT_REPETITION:
                break

        # If syncing all data then we must duplicate data dicts for each extra head
        if not self.training or random_num < self.INPUT_REPETITION:
            for head_id in range(1, len(self.head_datasets)):
                data_dict_arr.append(copy.deepcopy(data_dict_arr[0]))
                point_cloud_list.append(copy.deepcopy(data_dict_arr[0]['points']))

        ENABLE_VOXEL_CMB = 1

        # Combine at point cloud level and voxelize together
        if ENABLE_VOXEL_CMB == 0:
            # Overwrite the point cloud in the first head with the combined pointcloud
            data_dict_arr[0]['points'] = np.concatenate(point_cloud_list)
            # Send the combined point cloud to be shuffled and voxelized
            data_dict_arr[0] = self.data_processor_shffl_voxelize.forward(
                data_dict=data_dict_arr[0]
            )

        # Naive multi head approach, only voxelize one pointcloud to have one input
        if ENABLE_VOXEL_CMB == 2:
            # Send the first point cloud to be shuffled and voxelized
            data_dict_arr[0] = self.data_processor_shffl_voxelize.forward(
                data_dict=data_dict_arr[0]
            )

        # MIMO voxel combining
        if ENABLE_VOXEL_CMB == 1:
            # Store voxel information
            voxel_coords = []
            curr_voxel_index = 0

            # Calculate total number of voxels
            total_num_voxels = 0
            for head_id in range(self.NUM_HEADS):
                total_num_voxels += len(data_dict_arr[head_id]['voxel_coords'])

            # Now that we know the number of voxels we can make a numpy array and store directly to it
            voxels = np.zeros((total_num_voxels, self.MAX_POINTS_PER_VOXEL * self.NUM_HEADS, 5))
            # Init to all zeros since currently we haven't added points yet
            voxel_num_points = np.zeros(total_num_voxels, dtype=int)

            # First loop over to calculate number of voxels and create voxel coords
            # Create a dict with key=(x_coord,y_coord)
            # voxel_coords_set = set()
            voxel_coords_dict = {}
            for head_id in range(self.NUM_HEADS):
                for index, value in enumerate(data_dict_arr[head_id]['voxel_coords']):
                    xy_key = (value[1], value[2]) # value is [z_coord, x_coord, y_coord]
                    if xy_key not in voxel_coords_dict:
                        # Add the key to the dict
                        voxel_coords_dict[xy_key] = curr_voxel_index
                        curr_voxel_index += 1
                        # Append voxel coords
                        voxel_coords.append(data_dict_arr[head_id]['voxel_coords'][index])

                    # Add points to the voxel!
                    voxel_index = voxel_coords_dict[xy_key]
                    # Select only the valid points using number of points in voxel
                    num_points = data_dict_arr[head_id]['voxel_num_points'][index]

                    # Replace points from initial point to number of points to add
                    curr_num_points_in_voxel = voxel_num_points[voxel_index]

                    # Set first 5 rows to the points data
                    voxels[voxel_index][curr_num_points_in_voxel:curr_num_points_in_voxel + num_points][:,0:5] = \
                        data_dict_arr[head_id]['voxels'][index][:num_points]
                    # Set 5th column to head id
                    voxels[voxel_index][curr_num_points_in_voxel:curr_num_points_in_voxel + num_points][:,4] = head_id

                    # We already have the coords added but we have to update the point count
                    voxel_num_points[voxel_index] += num_points

            # SNIP to reduce size
            data_dict_arr[0]['voxels'] = voxels[:curr_voxel_index]
            data_dict_arr[0]['voxel_num_points'] = voxel_num_points[:curr_voxel_index]
            # Convert the other voxel information to np arrays
            data_dict_arr[0]['voxel_coords'] = np.array(voxel_coords)
            # Delete keys from other heads
            for i in range(1, self.NUM_HEADS):
                del data_dict_arr[i]['voxels']
                del data_dict_arr[i]['voxel_num_points']
                del data_dict_arr[i]['voxel_coords']

        return data_dict_arr

    def prepare_data(self, data_dict, head_dataset, head_id):
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
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(head_dataset.__len__())
                return head_dataset.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], head_dataset.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([head_dataset.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        # Add the head id as extra column to points
        new_column = np.full((len(data_dict['points']), 1), head_id)
        data_dict['points'] = np.hstack((data_dict['points'], new_column))

        data_dict = head_dataset.point_feature_encoder.forward(data_dict)

        # 1. mask_points_and_boxes_outside_range
        data_dict = self.data_processor_masking.forward(
            data_dict=data_dict
        )

        # Perform shuffle and voxel. Combine voxels later
        VOXEL_CMB = True
        if VOXEL_CMB:
            data_dict = self.data_processor_shffl_voxelize.forward(
                data_dict=data_dict
            )

        data_dict.pop('gt_names', None)

        return data_dict

    # This collate_batch function is modified from the one in dataset.py
    # Instead of receiving a list of data_dicts (N),
    # it receives a list of lists of data_dicts (N, number of heads)
    def collate_batch(self, batch_list, _unused=False):
        data_dict = defaultdict(list)
        batch_repetition = 1
        if self.training:
            batch_repetition = self.BATCH_REPETITION
        for i in range(batch_repetition):
            for cur_sample_list in batch_list:
                for cur_sample in cur_sample_list:
                    for key, val in cur_sample.items():
                        data_dict[key].append(val)
        # N * number of heads * batch repetition
        batch_size = len(batch_list) * len(batch_list[0]) * batch_repetition
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
        return self.head_datasets[0].generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path)

    # Must also use this method from one of our heads
    def evaluation(self, det_annos, class_names, **kwargs):
        print("mimo var evaluation called")
        print('len det annos', len(det_annos))
        return self.head_datasets[0].evaluation(det_annos, class_names, **kwargs)