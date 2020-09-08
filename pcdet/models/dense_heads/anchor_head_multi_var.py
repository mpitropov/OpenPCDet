import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .anchor_head_multi import SingleHead, AnchorHeadMulti

class SingleHeadVAR(SingleHead):
    def __init__(self, model_cfg, input_channels, *args, **kwargs):
        super().__init__(model_cfg, input_channels, *args, **kwargs)
        self.conv_var = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.code_size,
            kernel_size=1
        )
        nn.init.constant_(self.conv_var.weight, 0)
        nn.init.constant_(self.conv_var.bias, 0)

    def forward(self, spatial_features_2d):
        ret_dict = {}
        spatial_features_2d = super(SingleHead, self).forward({'spatial_features': spatial_features_2d})['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        var_preds = self.conv_var(spatial_features_2d)

        if self.separate_reg_config is None:
            box_preds = self.conv_box(spatial_features_2d)
        else:
            box_preds_list = []
            for reg_name in self.conv_box_names:
                box_preds_list.append(self.conv_box[reg_name](spatial_features_2d))
            box_preds = torch.cat(box_preds_list, dim=1)

        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
            var_preds = var_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = box_preds.view(-1, self.num_anchors_per_location,
                                       self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(-1, self.num_anchors_per_location,
                                       self.num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
            var_preds = var_preds.view(-1, self.num_anchors_per_location,
                                       self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            box_preds = box_preds.view(batch_size, -1, self.code_size)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
            var_preds = var_preds.view(batch_size, -1, self.code_size)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = dir_cls_preds.view(
                    -1, self.num_anchors_per_location, self.model_cfg.NUM_DIR_BINS, H, W).permute(0, 1, 3, 4,
                                                                                                  2).contiguous()
                dir_cls_preds = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        else:
            dir_cls_preds = None

        ret_dict['cls_preds'] = cls_preds
        ret_dict['box_preds'] = box_preds
        ret_dict['dir_cls_preds'] = dir_cls_preds
        ret_dict['var_preds'] = var_preds

        return ret_dict


class AnchorHeadMultiVAR(AnchorHeadMulti):
    def make_multihead(self, input_channels):
        rpn_head_cfgs = self.model_cfg.RPN_HEAD_CFGS
        rpn_heads = []
        class_names = []
        for rpn_head_cfg in rpn_head_cfgs:
            class_names.extend(rpn_head_cfg['HEAD_CLS_NAME'])

        for rpn_head_cfg in rpn_head_cfgs:
            num_anchors_per_location = sum([self.num_anchors_per_location[class_names.index(head_cls)]
                                            for head_cls in rpn_head_cfg['HEAD_CLS_NAME']])
            head_label_indices = torch.from_numpy(np.array([
                self.class_names.index(cur_name) + 1 for cur_name in rpn_head_cfg['HEAD_CLS_NAME']
            ]))

            rpn_head = SingleHeadVAR(
                self.model_cfg, input_channels,
                len(rpn_head_cfg['HEAD_CLS_NAME']) if self.separate_multihead else self.num_class,
                num_anchors_per_location, self.box_coder.code_size, rpn_head_cfg,
                head_label_indices=head_label_indices,
                separate_reg_config=self.model_cfg.get('SEPARATE_REG_CONFIG', None)
            )
            rpn_heads.append(rpn_head)
        self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        if self.shared_conv is not None:
            spatial_features_2d = self.shared_conv(spatial_features_2d)

        ret_dicts = []
        for rpn_head in self.rpn_heads:
            ret_dicts.append(rpn_head(spatial_features_2d))

        cls_preds = [ret_dict['cls_preds'] for ret_dict in ret_dicts]
        box_preds = [ret_dict['box_preds'] for ret_dict in ret_dicts]
        var_preds = [ret_dict['var_preds'] for ret_dict in ret_dicts]
        ret = {
            'cls_preds': cls_preds if self.separate_multihead else torch.cat(cls_preds, dim=1),
            'box_preds': box_preds if self.separate_multihead else torch.cat(box_preds, dim=1),
            'var_preds': var_preds if self.separate_multihead else torch.cat(var_preds, dim=1),
        }

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', False):
            dir_cls_preds = [ret_dict['dir_cls_preds'] for ret_dict in ret_dicts]
            ret['dir_cls_preds'] = dir_cls_preds if self.separate_multihead else torch.cat(dir_cls_preds, dim=1)

        self.forward_ret_dict.update(ret)

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds, batch_var_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=ret['cls_preds'], 
                box_preds=ret['box_preds'],
                var_preds=ret['var_preds'],
                dir_cls_preds=ret.get('dir_cls_preds', None)
            )

            if isinstance(batch_cls_preds, list):
                multihead_label_mapping = []
                for idx in range(len(batch_cls_preds)):
                    multihead_label_mapping.append(self.rpn_heads[idx].head_label_indices)

                data_dict['multihead_label_mapping'] = multihead_label_mapping

            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_var_preds'] = batch_var_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        
        loc_loss_weights = [
            losses_cfg.LOSS_WEIGHTS['loc_l1_weight'],
            losses_cfg.LOSS_WEIGHTS['loc_var_weight'],
            losses_cfg.LOSS_WEIGHTS['loc_calib_weight']
        ]
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(
                code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'],
                l1_weight=losses_cfg.LOSS_WEIGHTS['loc_l1_weight'],
                var_weight=losses_cfg.LOSS_WEIGHTS['loc_var_weight'],
                calib_weight=losses_cfg.LOSS_WEIGHTS['loc_calib_weight']
            )
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        var_preds = self.forward_ret_dict['var_preds']

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if not isinstance(box_preds, list):
            box_preds = [box_preds]
        batch_size = int(box_preds[0].shape[0])

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                     for anchor in self.anchors], dim=0
                )
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        start_idx = 0
        box_losses = 0
        tb_dict = {}
        for idx, box_pred in enumerate(box_preds):
            box_pred = box_pred.view(
                batch_size, -1,
                box_pred.shape[-1] // self.num_anchors_per_location if not self.use_multihead else box_pred.shape[-1]
            )
            box_reg_target = box_reg_targets[:, start_idx:start_idx + box_pred.shape[1]]
            reg_weight = reg_weights[:, start_idx:start_idx + box_pred.shape[1]]
            if box_dir_cls_preds is not None:
                take_sin_diff = True
            else:
                take_sin_diff = False
            loc_loss_src, l1_loss_src, var_loss_src, calib_loss_src = \
                self.reg_loss_func(box_pred, var_preds[idx], box_reg_target, anchors, \
                                   self.box_coder, take_sin_diff=take_sin_diff, weights=reg_weight)  # [N, M]

            loc_loss = loc_loss_src.sum() / batch_size
            l1_loss = l1_loss_src.sum() / batch_size
            var_loss = var_loss_src.sum() / batch_size
            calib_loss = calib_loss_src.sum() / batch_size

            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            box_losses += loc_loss
            tb_dict['rpn_loss_loc'] = tb_dict.get('rpn_loss_loc', 0) + loc_loss.item()
            tb_dict['rpn_loss_l1'] = tb_dict.get('rpn_loss_l1', 0) + l1_loss.item()
            tb_dict['rpn_loss_var'] = tb_dict.get('rpn_loss_var', 0) + var_loss.item()
            tb_dict['rpn_loss_calib'] = tb_dict.get('rpn_loss_calib', 0) + calib_loss.item()

            if box_dir_cls_preds is not None:
                if not isinstance(box_dir_cls_preds, list):
                    box_dir_cls_preds = [box_dir_cls_preds]
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )
                box_dir_cls_pred = box_dir_cls_preds[idx]
                dir_logit = box_dir_cls_pred.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                weights = positives.type_as(dir_logit)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)

                weight = weights[:, start_idx:start_idx + box_pred.shape[1]]
                dir_target = dir_targets[:, start_idx:start_idx + box_pred.shape[1]]
                dir_loss = self.dir_loss_func(dir_logit, dir_target, weights=weight)
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                box_losses += dir_loss
                tb_dict['rpn_loss_dir'] = tb_dict.get('rpn_loss_dir', 0) + dir_loss.item()
            start_idx += box_pred.shape[1]
        return box_losses, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, var_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        batch_var_preds = var_preds.view(batch_size, num_anchors, -1) if not isinstance(var_preds, list) \
            else torch.cat(var_preds, dim=1).view(batch_size, num_anchors, -1)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds, batch_var_preds