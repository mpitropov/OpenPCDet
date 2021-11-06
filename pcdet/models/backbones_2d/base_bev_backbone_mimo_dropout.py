import numpy as np
import torch
from torch._C import TensorType
import torch.nn as nn
import copy

class BaseBEVBackboneMIMODropout(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        # MIMO specific params
        if self.model_cfg.get('NUM_HEADS', None) is not None:
            self.NUM_HEADS = self.model_cfg.NUM_HEADS
        if self.model_cfg.get('INPUT_REPETITION', None) is not None:
            self.INPUT_REPETITION = self.model_cfg.INPUT_REPETITION
        if self.model_cfg.get('BATCH_REPETITION', None) is not None:
            self.BATCH_REPETITION = self.model_cfg.BATCH_REPETITION
        self.rng = np.random.default_rng()

        # Multiply by NUM_HEADS for MIMO
        input_channels = input_channels * self.NUM_HEADS

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Dropout(0.5),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # MIMO specific code
        # Stack spatial features and add gt to the input
        batch_size = data_dict['batch_size']
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

        gt_boxes = [] # This is simply an array of GTs to add
        spatial_feats = [] # This is a bit more complex each row is a head and columns are things to add to that head
        for frame_group_index in range(len(frame_list)):
            new_spatial_feats_row = []
            # Loop through frames in this grouping
            for head_id in range(self.NUM_HEADS):
                # Get the batch index to copy over information
                batch_list_index = frame_list[frame_group_index][head_id]
                gt_boxes.append(data_dict['gt_boxes'][batch_list_index])
                # Don't clone first head
                if head_id == 0:
                    new_spatial_feats_row.append(data_dict['spatial_features'][batch_list_index])
                else:
                    new_spatial_feats_row.append(data_dict['spatial_features'][batch_list_index].clone())
            # Add to the original list
            test = torch.cat(new_spatial_feats_row, 0).unsqueeze(0)
            spatial_feats.append(test)

        # N * number of heads * batch repetition
        data_dict['batch_size'] = data_dict['batch_size'] * self.NUM_HEADS * batch_repetitions
        data_dict['gt_boxes'] = torch.stack(gt_boxes)
        data_dict['spatial_features'] = torch.stack(spatial_feats, 1).squeeze(0)

        # Original code
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
