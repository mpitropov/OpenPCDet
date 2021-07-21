import numpy as np
import torch
import torch.nn as nn

from .anchor_head_single_var import AnchorHeadSingleVAR

class AnchorHeadMIMO(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__()
        self.predict_boxes_when_training = predict_boxes_when_training
        self.NUM_HEADS = model_cfg.NUM_HEADS

        # Create detection heads
        self.detection_heads = nn.ModuleList()
        for i in range(self.NUM_HEADS):
            self.detection_heads.append(
                AnchorHeadSingleVAR(model_cfg, input_channels, num_class,
                                    class_names, grid_size, point_cloud_range,
                                    predict_boxes_when_training=True))

        # Replaced on first forward pass
        self.batch_size = 0
        self.head_gt_indices = []

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        gt_boxes = None
        if 'gt_boxes' in data_dict:
            gt_boxes = data_dict['gt_boxes']

        # Calculate batch size based on batch_size and number of heads
        # batch size might be different on last epoch
        tmp_batch_size = int(data_dict['batch_size'] / self.NUM_HEADS)
        if self.batch_size != tmp_batch_size:
            self.batch_size = tmp_batch_size
            # Pass on specific gt_boxes to each head
            # Example num_head=3, batch_size=2 creates a=[0,3], b=[1,4], c=[2,5]
            self.head_gt_indices = []
            for i in range(self.NUM_HEADS):
                self.head_gt_indices.append( \
                    torch.tensor(np.arange(i, self.NUM_HEADS*self.batch_size, self.NUM_HEADS)).cuda())

        # Run forward pass of each head
        ret_data_dicts = []
        for i in range(self.NUM_HEADS):
            selected_gt_boxes = None
            if 'gt_boxes' in data_dict:
                selected_gt_boxes = gt_boxes.index_select(0, self.head_gt_indices[i])
            ret_data_dicts.append(
                self.detection_heads[i]({
                    'spatial_features_2d': spatial_features_2d,
                    'gt_boxes': selected_gt_boxes,
                    'batch_size': self.batch_size
                })
            )
            # Store batch features in data_dict
            if i == 0:
                data_dict['batch_features'] = ret_data_dicts[i]['batch_features']
            else:
                data_dict['batch_features_' + str(i)] = ret_data_dicts[i]['batch_features']

        if not self.training or self.predict_boxes_when_training:
            for i in range(self.NUM_HEADS):
                if i == 0:
                    data_dict['batch_cls_preds'] = ret_data_dicts[i]['batch_cls_preds']
                    data_dict['batch_box_preds'] = ret_data_dicts[i]['batch_box_preds']
                    data_dict['batch_var_preds'] = ret_data_dicts[i]['batch_var_preds']
                    data_dict['cls_preds_normalized'] = ret_data_dicts[i]['cls_preds_normalized']
                else:
                    data_dict['batch_cls_preds_' + str(i)] = ret_data_dicts[i]['batch_cls_preds']
                    data_dict['batch_box_preds_' + str(i)] = ret_data_dicts[i]['batch_box_preds']
                    data_dict['batch_var_preds_' + str(i)] = ret_data_dicts[i]['batch_var_preds']
                    data_dict['cls_preds_normalized_' + str(i)] = ret_data_dicts[i]['cls_preds_normalized']

        return data_dict

    def get_loss(self):
        # Get RPN loss and tensorboard output from each head
        rpn_loss_list = []
        tb_dict_list = []
        for i in range(self.NUM_HEADS):
            tmp_rpn_loss, tmp_tb_dict = self.detection_heads[i].get_loss()
            rpn_loss_list.append(tmp_rpn_loss)
            tb_dict_list.append(tmp_tb_dict)

        # Store individual head outputs
        rpn_loss_l1_list = []
        rpn_loss_cls_list = []
        for i in range(self.NUM_HEADS):
            rpn_loss_l1_list.append(tb_dict_list[i]['rpn_loss_l1'])
            rpn_loss_cls_list.append(tb_dict_list[i]['rpn_loss_cls'])

        # Average tensorboard outputs from all heads
        for key in tb_dict_list[0]:
            for i in range(1, self.NUM_HEADS):
                tb_dict_list[0][key] += tb_dict_list[i][key]
            tb_dict_list[0][key] /= self.NUM_HEADS

        # Now add the individual head outputs
        for i in range(self.NUM_HEADS):
            tb_dict_list[0]['rpn_loss_l1_h' + str(i)] = rpn_loss_l1_list[i]
            tb_dict_list[0]['rpn_loss_cls_h' + str(i)] = rpn_loss_cls_list[i]

        # Add rpn losses together
        rpn_loss = 0
        for i in range(self.NUM_HEADS):
            rpn_loss += rpn_loss_list[i]
        rpn_loss = rpn_loss / self.NUM_HEADS

        return rpn_loss, tb_dict_list[0]
