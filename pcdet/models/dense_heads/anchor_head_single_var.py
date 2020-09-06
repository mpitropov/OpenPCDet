import numpy as np
import torch.nn as nn

from .anchor_head_single import AnchorHeadSingle


class AnchorHeadSingleVAR(AnchorHeadSingle):
    def __init__(self, model_cfg, input_channels, *args, **kwargs):
        super().__init__(model_cfg, input_channels, *args, **kwargs)
        self.conv_var = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.code_size,
            kernel_size=1
        )
        nn.init.constant_(self.conv_var.weight, 0)
        nn.init.constant_(self.conv_var.bias, 0)


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        var_preds = self.conv_var(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        var_preds = var_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['var_preds'] = var_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds, batch_var_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, 
                box_preds=box_preds, 
                var_preds=var_preds,
                dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_var_preds'] = batch_var_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
