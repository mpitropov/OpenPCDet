import torch

from .pointpillar import PointPillar
from ..model_utils import model_nms_utils

class PointPillarMIMOVAR(PointPillar):
    def post_processing(self, batch_dict):
        """
        For MIMO call post processing on each head then use IoU to combine
        """
        # OVERRIDE FOR MIMO testing
        batch_dict['batch_size'] = 1

        pred_dicts, recall_dict = post_processing_single(batch_dict)

        return pred_dicts, recall_dict

    def post_processing_single(self, batch_dict):
        """
        Instead of returning final boxes after NMS, this post processing
        function returns all anchor boxes along with selected indices
        indicating the selected boxes.
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                batch_var_preds: (B, num_boxes, 1) or (N1+N2+..., 1+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                clf_loss_name = self.model_cfg.DENSE_HEAD.LOSS_CONFIG.get('CLF_LOSS_TYPE', 'SigmoidFocalClassificationLoss')
                if clf_loss_name == 'SigmoidFocalClassificationLoss':
                    num_classes = self.num_class
                elif clf_loss_name == 'SoftmaxFocalLossV1' or \
                    clf_loss_name == 'SoftmaxFocalLossV2':
                    num_classes = self.num_class + 1
                assert cls_preds.shape[1] in [1, num_classes]

                if not batch_dict['cls_preds_normalized']:
                    if clf_loss_name == 'SigmoidFocalClassificationLoss':
                        cls_preds = torch.sigmoid(cls_preds)
                    elif clf_loss_name == 'SoftmaxFocalLossV1' or \
                        clf_loss_name == 'SoftmaxFocalLossV2':
                        # Perform softmax and remove background class
                        cls_preds = torch.softmax(cls_preds, dim=-1)[...,:-1]
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    if clf_loss_name == 'SigmoidFocalClassificationLoss':
                        cls_preds = [torch.sigmoid(x) for x in cls_preds]
                    elif clf_loss_name == 'SoftmaxFocalLossV1' or \
                        clf_loss_name == 'SoftmaxFocalLossV2':
                        # TODO: Implement for multi head
                        raise NotImplementedError

            var_preds = batch_dict['batch_var_preds'][batch_mask]
            var_preds = torch.exp(var_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                cur_offset_idx = 0
                pred_scores, pred_labels, pred_boxes, pred_vars, selected = [], [], [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_var_preds = var_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes, cur_pred_vars, cur_selected = model_nms_utils.multi_classes_nms_var(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds, var_preds=cur_var_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    pred_vars.append(cur_pred_vars)
                    selected.append(cur_selected + cur_offset_idx)
                    cur_start_idx += cur_cls_preds.shape[0]
                    cur_offset_idx += cur_pred_scores.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
                final_vars = torch.cat(pred_vars, dim=0)
                final_selected = torch.cat(selected, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = cls_preds
                final_labels = label_preds
                final_boxes = box_preds
                final_vars = var_preds
                final_selected = selected

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )


            record_dict = {
                'feature': batch_dict['batch_features'][index],
                'pred_boxes': final_boxes[final_selected],
                'pred_scores': final_scores[final_selected],
                'pred_scores_all': src_cls_preds[final_selected],
                'pred_labels': final_labels[final_selected],
                'pred_vars': final_vars[final_selected],
                'anchor_boxes': final_boxes,
                'anchor_scores': final_scores,
                'anchor_labels': final_labels,
                'anchor_vars': final_vars,
                'selected': final_selected
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

# Martin Ma - 2020/08/14
import numpy as np
from math import pi, cos, sin

def preprocess(model_outputs, iouThresh=0.2):
    """ Preprocess the model_outputs to the format that the acquisition 
        function requires
    
    Args:
        model_outputs (a list of dicts of list)

    Return:
        a dictionary which contains the names, scores, boxes, boxes_var and box_mean
        grouped by objects and by frames.
        **It is crucial to realize that the boxes_var is not the variance of all boxes,
        rather, it is given directly from the model**
    """
    names, scores, boxes_lidar, boxes_lidar_var = extract_info(model_outputs)
    grouped_names, grouped_scores, grouped_boxes, grouped_boxes_var = \
    grouping(names, scores, boxes_lidar, boxes_lidar_var, iouThresh)

    results = {
        "names": grouped_names,
        "scores": grouped_scores,
        "boxes": grouped_boxes,
        "boxes_var": grouped_boxes_var,
    }
    return results
    

def extract_info(model_outputs):
    """ Extract info to the format that can be fed to downstreams (e.x. acquisition function)
    
    Args:
        model_outputs (a list of dicts of lists)

    Returns:
        names (a list of lists): shape -> (# of samples, # of objects in a sample)
        scores (a list of lists): shape -> (# of samples, # of objects in a sample)
        boxes_lidar (a list of lists of lists): shape -> (# of samples, # of objects in a sample,
                                                            # of dimension in a box)
    """
    names, scores, boxes_lidar, boxes_lidar_var = [], [], [], []
    
    # Handle edge cases
    if len(model_outputs) == 0:
        return names, scores, boxes_lidar, boxes_lidar_var
    if type(model_outputs) == dict:
        names.append(model_outputs['name'])
        scores.append(model_outputs['score'])
        boxes_lidar.append(model_outputs['boxes_lidar'])
        boxes_lidar_var.append(model_outputs['boxes_lidar_var'])
        return names, scores, boxes_lidar, boxes_lidar_var

    for model_output in model_outputs:
        names.append(model_output['name'])
        scores.append(model_output['score'])
        boxes_lidar.append(model_output['boxes_lidar'])
        boxes_lidar_var.append(model_output['boxes_lidar_var'])
    return names, scores, boxes_lidar, boxes_lidar_var

def grouping(names, scores, boxes, boxes_var, iouThresh):
    """ Group the detections (multiple MC passes) by objects for multiple samples.
        This function is able to handle 3d bbox or bev box. However,
        for 3d bbox, it doesn't consider the z-axis, making it analogous to the bev case
    
    Args:
        boxes (a list of numpy array): 
            shape -> (batch_size, # of objects in a sample,
                        # of dimension in a detection)
        scores
        boxes
        boxes_var
        iouThresh (float): boxes with iou higher than iouThresh are considered
                           to belong to the same object
    
    Returns:
        grouped_boxes (list of lists of numpy array): 
            shape -> (batch_size, # of objects in a sample,
                        # of detections in an object (varies),
                        # of dimension in a detection)
        grouped_names
        grouped_scores
        grouped_boxes_var
    """
    grouped_names, grouped_scores, grouped_boxes, grouped_boxes_var = [], [], [], []

    for i in range(len(boxes)):
        clusters, _ = grouping_single_sample(boxes[i][:,[0,1,3,4,6]], iouThresh)
        grouped_names_single_sample, grouped_scores_single_sample, grouped_boxes_single_sample, \
        grouped_boxes_var_single_sample = [], [], [], []
        for cluster in clusters:
            grouped_names_single_sample.append(np.array(names[i][cluster]))
            grouped_scores_single_sample.append(np.array(scores[i][cluster]))
            grouped_boxes_single_sample.append(np.array(boxes[i][cluster, :]))
            grouped_boxes_var_single_sample.append(np.exp(np.array(boxes[i][cluster, :])))
        grouped_names.append(grouped_names_single_sample)
        grouped_scores.append(grouped_scores_single_sample)
        grouped_boxes.append(grouped_boxes_single_sample)
        grouped_boxes_var.append(grouped_boxes_var_single_sample)

    return grouped_names, grouped_scores, grouped_boxes, grouped_boxes_var

def grouping_single_sample(boxes, iouThresh):
    """ Group the detections (multiple MC passes) by objects for a single sample.
        This function handle bev coordinates (x, y, w, h, rotation)

    Args:
        boxes (numpy array): shape -> (# of objects in a sample,
                                       # of dimension in a detection = 5)
    
    Returns:
        clusters (a list of list):
            first list is of size = # of groups (objects)
            second (nested) list is of size = # of detection for that group (object)
            the elements represent the index of the box that belongs to the ith group
        cluster_mean (a list of list):
            first list is the of size = # of groups (objects)
            second (nested) list is the mean of all detections in that group
    """
    boxes = np.array(boxes)
    
    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    clusters = []
    clusters_mean = []
    
    for i in range(len(boxes)):
        matched_to_cluster = False
        for j in range(len(clusters)):
            # r1 and r2 are in (center, width, height, rotation) representation
            # First convert these into a sequence of vertices
            box_rect_coord = rectangle_vertices(*boxes[i])
            mean_rect_coord = rectangle_vertices(*clusters_mean[j])

            # Detemine if IoU meets the threshold
            intersection = intersection_area(box_rect_coord, mean_rect_coord)
            union = area(box_rect_coord) + area(mean_rect_coord) - intersection
            iou = intersection / union
            if iou >= iouThresh:
                matched_to_cluster = True
                clusters[j].append(i)
                clusters_mean[j] = recalculate_cluster_mean(boxes[i], len(clusters[j]), clusters_mean[j])
                break
        if matched_to_cluster == False:
            clusters.append([i])
            clusters_mean.append(boxes[i])
        
    return clusters, clusters_mean

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )

def rectangle_vertices(cx, cy, w, h, r):
    angle = pi*r/180
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )

def intersection_area(rect1, rect2):
    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))

def area(rectangle):
    result = abs(0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(rectangle, rectangle[1:]+rectangle[:1])))
    return result

def recalculate_cluster_mean(current_box, num_cluster, cluster_mean):
    current_box = np.array(current_box)
    cluster_mean = np.array(cluster_mean)
    return current_box / num_cluster + cluster_mean * (num_cluster-1) / num_cluster
