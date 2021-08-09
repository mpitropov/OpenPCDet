import copy
import numpy as np
from collections import defaultdict

from . import common_utils

# MIMO Type A: During training the voxelized point clouds are combined together
def modify_batch_a(self, batch_list):
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
        # Store voxel information
        voxel_coords = []
        curr_voxel_index = 0
        total_num_voxels = 0

        # Loop through frames in this grouping
        for head_id in range(self.NUM_HEADS):
            batch_list_index = frame_list[frame_group_index][head_id]
            # Add non voxel components
            for key, val in batch_list[batch_list_index].items():
                if key in ['voxels', 'voxel_coords', 'voxel_num_points']:
                    continue
                data_dict[key].append(val)
            # Calculate total number of voxels
            total_num_voxels += len(batch_list[batch_list_index]['voxel_coords'])

        # Now that we know the number of voxels we can make a numpy array and store directly to it
        NUM_POINT_FEATURES = len(self.dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)
        voxels = np.zeros((total_num_voxels, self.MAX_POINTS_PER_VOXEL * self.NUM_HEADS, NUM_POINT_FEATURES))
        # Init to all zeros since currently we haven't added points yet
        voxel_num_points = np.zeros(total_num_voxels, dtype=int)

        # First loop over to calculate number of voxels and create voxel coords
        # Create a dict with key=(z_coord, x_coord, y_coord)
        voxel_coords_dict = {}
        for head_id in range(self.NUM_HEADS):
            batch_list_index = frame_list[frame_group_index][head_id]

            for index, value in enumerate(batch_list[batch_list_index]['voxel_coords']):
                zxy_key = (value[0], value[1], value[2]) # value is [z_coord, x_coord, y_coord]
                if zxy_key not in voxel_coords_dict:
                    # Add the key to the dict
                    voxel_coords_dict[zxy_key] = curr_voxel_index
                    curr_voxel_index += 1
                    # Append voxel coords
                    voxel_coords.append(batch_list[batch_list_index]['voxel_coords'][index])

                # Add points to the voxel!
                voxel_index = voxel_coords_dict[zxy_key]
                # Select only the valid points using number of points in voxel
                num_points = batch_list[batch_list_index]['voxel_num_points'][index]

                # Replace points from initial point to number of points to add
                curr_num_points_in_voxel = voxel_num_points[voxel_index]

                # Set first X columns to the points data
                voxels[voxel_index][curr_num_points_in_voxel:curr_num_points_in_voxel + num_points][:,0:NUM_POINT_FEATURES-1] = \
                    batch_list[batch_list_index]['voxels'][index][:num_points]
                # Set the last column to head id
                voxels[voxel_index][curr_num_points_in_voxel:curr_num_points_in_voxel + num_points][:,NUM_POINT_FEATURES-1] = head_id
                # We already have the coords added but we have to update the point count
                voxel_num_points[voxel_index] += num_points

        # SNIP to reduce size
        data_dict['voxels'].append(voxels[:curr_voxel_index])
        data_dict['voxel_num_points'].append(voxel_num_points[:curr_voxel_index])
        # Convert the other voxel information to np arrays
        data_dict['voxel_coords'].append(np.array(voxel_coords))

    return data_dict

# MIMO type B: Custom prepare data function
# Which masks points but only performs shuffle & voxelize during testing
def prepare_data_b(self, data_dict, head_dataset):
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

    data_dict = self.data_processor_masking.forward(
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

# MIMO Type B: During training the pointclouds must be combined and voxelized together
def modify_batch_b(self, batch_list):
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
        tmp_dict = {
            'points': points,
            'use_lead_xyz': data_dict['use_lead_xyz'][frame_group_index]
        }

        # Shuffle and voxelize the pointcloud
        tmp_dict = self.data_processor_shffl_voxelize.forward(
            data_dict=tmp_dict
        )

        # Add the voxel stuff to the data_dict
        data_dict['voxels'].append(tmp_dict['voxels'])
        data_dict['voxel_coords'].append(tmp_dict['voxel_coords'])
        data_dict['voxel_num_points'].append(tmp_dict['voxel_num_points'])

    return data_dict