import numpy as np
import torch
import torch.nn as nn

from .anchor_head_single_var import AnchorHeadSingleVAR

class AnchorHeadMIMO(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__()
        self.predict_boxes_when_training = predict_boxes_when_training
        self.num_heads = 3

        # Create detection heads
        detection_heads = []
        for i in range(self.num_heads):
            detection_heads.append(
                AnchorHeadSingleVAR(model_cfg, input_channels, num_class,
                                    class_names, grid_size, point_cloud_range,
                                    predict_boxes_when_training=True))
        self.head_a = detection_heads[0]
        self.head_b = detection_heads[1]
        self.head_c = detection_heads[2]


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        gt_boxes = data_dict['gt_boxes']
        batch_size = int(data_dict['batch_size'] / self.num_heads)

        # Pass on specific gt_boxes to each head
        if batch_size == 1:
            head_a_gt_indices = torch.tensor([0]).cuda()
            head_b_gt_indices = torch.tensor([1]).cuda()
            head_c_gt_indices = torch.tensor([2]).cuda()
        elif batch_size == 2:
            head_a_gt_indices = torch.tensor([0, 3]).cuda()
            head_b_gt_indices = torch.tensor([1, 4]).cuda()
            head_c_gt_indices = torch.tensor([2, 5]).cuda()
        else:
            print("ERROR BAD BATCH SIZE")
            exit()

        # Run forward pass of each head
        ret_data_dict_a = self.head_a({
            'spatial_features_2d': spatial_features_2d,
            'gt_boxes': gt_boxes.index_select(0, head_a_gt_indices),
            'batch_size': batch_size
        })
        ret_data_dict_b = self.head_b({
            'spatial_features_2d': spatial_features_2d,
            'gt_boxes': gt_boxes.index_select(0, head_b_gt_indices),
            'batch_size': batch_size
        })
        ret_data_dict_c = self.head_c({
            'spatial_features_2d': spatial_features_2d,
            'gt_boxes': gt_boxes.index_select(0, head_c_gt_indices),
            'batch_size': batch_size
        })

        data_dict['batch_features'] = ret_data_dict_a['batch_features']
        data_dict['batch_features_b'] = ret_data_dict_b['batch_features']
        data_dict['batch_features_c'] = ret_data_dict_c['batch_features']

        if not self.training or self.predict_boxes_when_training:
            data_dict['batch_cls_preds'] = ret_data_dict_a['batch_cls_preds']
            data_dict['batch_box_preds'] = ret_data_dict_a['batch_box_preds']
            data_dict['batch_var_preds'] = ret_data_dict_a['batch_var_preds']
            data_dict['cls_preds_normalized'] = ret_data_dict_a['cls_preds_normalized']
            data_dict['batch_cls_preds_b'] = ret_data_dict_b['batch_cls_preds']
            data_dict['batch_box_preds_b'] = ret_data_dict_b['batch_box_preds']
            data_dict['batch_var_preds_b'] = ret_data_dict_b['batch_var_preds']
            data_dict['cls_preds_normalized_b'] = ret_data_dict_b['cls_preds_normalized']
            data_dict['batch_cls_preds_c'] = ret_data_dict_c['batch_cls_preds']
            data_dict['batch_box_preds_c'] = ret_data_dict_c['batch_box_preds']
            data_dict['batch_var_preds_c'] = ret_data_dict_c['batch_var_preds']
            data_dict['cls_preds_normalized_c'] = ret_data_dict_c['cls_preds_normalized']

        return data_dict

    def get_loss(self):
        # Get RPN loss and tensorboard output from each head
        rpn_loss_a, tb_dict = self.head_a.get_loss()
        rpn_loss_b, tb_dict_b = self.head_b.get_loss()
        rpn_loss_c, tb_dict_c = self.head_c.get_loss()

        # Add tensorboard outputs from all three heads
        for key in tb_dict:
            tb_dict[key] += tb_dict_b[key]
            tb_dict[key] += tb_dict_c[key]
            tb_dict[key] /= self.num_heads

        # Add rpn losses together
        rpn_loss = (rpn_loss_a + rpn_loss_b + rpn_loss_c) / self.num_heads

        return rpn_loss, tb_dict
