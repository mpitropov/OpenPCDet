import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.von_mises import _log_modified_bessel_fn

from . import box_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

class SoftmaxFocalLossV1(nn.Module):
    """
    Softmax focal cross entropy loss.
    Implementation from:
    https://gluon-cv.mxnet.io/_modules/gluoncv/loss.html#FocalLoss
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SoftmaxFocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        # Compute softmax on classes, then remove the background class
        pred_softmax = torch.softmax(input, dim=2)[...,:-1]
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_softmax) + (1.0 - target) * pred_softmax
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = F.binary_cross_entropy(pred_softmax, target, reduction='none')

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

class SoftmaxFocalLossV2(nn.Module):
    """
    Softmax focal cross entropy loss.
    Implementation from function SoftmaxFocalLossKernel:
    https://github.com/pytorch/pytorch/blob/master/modules/detectron/softmax_focal_loss_op.cu
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SoftmaxFocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_softmax = torch.softmax(input, dim=2)
        alpha_weight = pred_softmax.new_full(pred_softmax.shape, self.alpha)
        alpha_weight[...,-1] = 1.0 - self.alpha
        focal_weight = alpha_weight * torch.pow(1.0 - pred_softmax, self.gamma)

        ce_loss = -target * torch.log(torch.clamp(pred_softmax, min=1e-10))

        loss = focal_weight * ce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


class VarRegLoss(nn.Module):
    """
    Calculate loss for the regression log variance output.
    """
    def __init__(self, 
                 beta: float = 1.0 / 9.0,
                 code_weights: list = None,
                 l1_weight: float = 0.0,
                 var_weight: float = 1.0):
        super(VarRegLoss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        self.l1_weight = l1_weight
        self.var_weight = var_weight

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def add_cos_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.cos(boxes1[..., dim:dim + 1] * 2) * torch.cos(boxes2[..., dim:dim + 1] * 2)
        rad_tg_encoding = torch.sin(boxes1[..., dim:dim + 1] * 2) * torch.sin(boxes2[..., dim:dim + 1] * 2)
        return rad_pred_encoding, rad_tg_encoding

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, reg_preds: torch.Tensor, var_preds: torch.Tensor, gt_targets: torch.Tensor, \
        anchors: torch.Tensor, box_coder, take_sin_diff: bool, weights: torch.Tensor):
        """
        Args:
            reg_preds: (B, #anchors, 7) float tensor.
                Predicted regression values for each anchor.
            gt_targets: (B, #anchors, 7) float tensor.
                Regression value targets.
            var_preds: (B, #anchors, 7) float tensor.
                Predicted regression value log variances for each anchor.

        Returns:
            loss: (B, #anchors, 7) float tensor.
                loss for each anchor and 7 respective log variances
        """

        if torch.isnan(gt_targets).any():
            print(" first gt_targets has nan")
            exit()
        if torch.isinf(gt_targets).any():
            print(" first gt_targets has inf")
            exit()

        # gt_targets = torch.where(torch.isnan(gt_targets), reg_preds, gt_targets)  # ignore nan targets
        reg_preds = torch.where(torch.isnan(reg_preds), gt_targets, reg_preds)  # ignore nan targets

        if torch.isnan(reg_preds).any():
            print(" reg_preds has nan")
            exit()
        if torch.isinf(reg_preds).any():
            print(" reg_preds has inf")
            exit()


        if torch.isnan(gt_targets).any():
            print(" gt_targets has nan")
            exit()
        if torch.isinf(gt_targets).any():
            print(" gt_targets has inf")
            exit()

        # sin(a - b) = sinacosb-cosasinb
        if take_sin_diff:
            reg_preds_sin, gt_targets_sin = self.add_sin_difference(reg_preds, gt_targets)
            zero_tensor = torch.zeros(reg_preds_sin.size()).cuda()
            reg_preds_sin = torch.where(torch.isnan(reg_preds_sin), zero_tensor, reg_preds_sin)
            gt_targets_sin = torch.where(torch.isnan(gt_targets_sin), zero_tensor, gt_targets_sin)
            if torch.isnan(reg_preds_sin).any():
                print(" reg_preds_sin has nan")
                exit()
            if torch.isinf(reg_preds_sin).any():
                print(" reg_preds_sin has inf")
                exit()
            if torch.isnan(gt_targets_sin).any():
                print(" gt_targets_sin has nan")
                exit()
            if torch.isinf(gt_targets_sin).any():
                print(" gt_targets_sin has inf")
                exit()
            diff = gt_targets_sin - reg_preds_sin
        else:
            # Not used (does not converge)
            diff = gt_targets - reg_preds

        # cos(2a - 2b) = cos2acos2b+sin2asin2b
        reg_preds_cos, gt_targets_cos = self.add_cos_difference(reg_preds, gt_targets)
        var_angle_diff = (gt_targets_cos + reg_preds_cos).squeeze(-1)

        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)
            var_angle_diff = var_angle_diff * self.code_weights.view(1, 1, -1)[...,6]

        var_preds = torch.clamp(var_preds, min=-10, max=10)

        zero_tensor = torch.zeros(var_preds.size()).cuda()
        var_preds = torch.where(torch.isnan(var_preds), zero_tensor, var_preds)
        diff_decoded = torch.where(torch.isnan(diff_decoded), zero_tensor, diff_decoded)

        if torch.isnan(diff).any():
            print(" diff has nan")
            exit()
        if torch.isinf(diff).any():
            print(" diff has inf")
            exit()
        if torch.isnan(var_preds).any():
            print(" var_preds has nan")
            exit()
        if torch.isinf(var_preds).any():
            print(" var_preds has inf")
            exit()
        if torch.isnan(diff_decoded).any():
            print(" diff_decoded has nan")
            exit()
        if torch.isinf(diff_decoded).any():
            print(" diff_decoded has inf")
            exit()

        zero_tensor = torch.zeros(var_preds.size()).cuda()
        var_preds = torch.where(torch.isnan(var_preds), zero_tensor, var_preds)
        diff_decoded = torch.where(torch.isnan(diff_decoded), zero_tensor, diff_decoded)

        if torch.isnan(diff).any():
            print(" diff has nan")
            exit()
        if torch.isinf(diff).any():
            print(" diff has inf")
            exit()
        if torch.isnan(var_preds).any():
            print(" var_preds has nan")
            exit()
        if torch.isinf(var_preds).any():
            print(" var_preds has inf")
            exit()
        if torch.isnan(diff_decoded).any():
            print(" diff_decoded has nan")
            exit()
        if torch.isinf(diff_decoded).any():
            print(" diff_decoded has inf")
            exit()

        loss_l1 = self.smooth_l1_loss(diff, self.beta)
        loss_var_linear = 0.5*(torch.exp(-var_preds[..., :6])*torch.pow(diff[..., :6], 2)) + \
                    0.5*var_preds[..., :6]
        s0 = 1.0 # Offset
        # TODO: Replace with torch.log(torch.i0()) when the gradient is implemented
        loss_var_angle = _log_modified_bessel_fn(torch.exp(-var_preds[..., 6]), order=0) - \
                            torch.exp(-var_preds[..., 6]) * var_angle_diff + \
                            F.elu(var_preds[..., 6] - s0)
        loss_var = torch.cat([loss_var_linear, loss_var_angle.unsqueeze(-1)], dim=-1)

        loss_l1 = torch.where(torch.isnan(loss_l1), zero_tensor, loss_l1)
        loss_var = torch.where(torch.isnan(loss_var), zero_tensor, loss_var)
        loss_calib = torch.where(torch.isnan(loss_calib), zero_tensor, loss_calib)

        if torch.isnan(loss_l1).any():
            print(" loss_l1 has nan")
            exit()
        if torch.isinf(loss_l1).any():
            print(" loss_l1 has inf")
            exit()
            
        if torch.isnan(loss_var).any():
            print(" loss_var has nan")
            exit()
        if torch.isinf(loss_var).any():
            print(" loss_var has inf")
            exit()

        if torch.isnan(loss_calib).any():
            print(" loss_calib has nan")
            exit()
        if torch.isinf(loss_calib).any():
            print(" loss_calib has inf")
            exit()

        loss_l1 = torch.where(torch.isnan(loss_l1), zero_tensor, loss_l1)
        loss_var = torch.where(torch.isnan(loss_var), zero_tensor, loss_var)
        loss_calib = torch.where(torch.isnan(loss_calib), zero_tensor, loss_calib)

        if torch.isnan(loss_l1).any():
            print(" loss_l1 has nan")
            exit()
        if torch.isinf(loss_l1).any():
            print(" loss_l1 has inf")
            exit()
            
        if torch.isnan(loss_var).any():
            print(" loss_var has nan")
            exit()
        if torch.isinf(loss_var).any():
            print(" loss_var has inf")
            exit()

        if torch.isnan(loss_calib).any():
            print(" loss_calib has nan")
            exit()
        if torch.isinf(loss_calib).any():
            print(" loss_calib has inf")
            exit()

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss_l1.shape[0] and weights.shape[1] == loss_l1.shape[1]
            loss_l1 *= weights.unsqueeze(-1)
            loss_var *= weights.unsqueeze(-1)
            loss_var_linear *= weights.unsqueeze(-1)
            loss_var_angle *= weights

        loss = self.l1_weight*loss_l1 + self.var_weight*loss_var

        return loss, loss_l1.clone().detach(), loss_var.clone().detach(), \
                loss_var_linear.clone().detach(), loss_var_angle.clone().detach()
