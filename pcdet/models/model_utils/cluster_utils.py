# ANONYMOUS - 2020/08/14

# NOTE: This can be used to cluster the outputs from MIMO heads
# We currently don't use it because we evaluate with the UncertaintyEval code
# instead of the code withn OpenPCDet.

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
    names, scores, scores_all, boxes_lidar, boxes_lidar_var, head_ids = extract_info(model_outputs)
    grouped_names, grouped_scores, grouped_scores_all, grouped_boxes, grouped_boxes_var, grouped_head_ids = \
    grouping(names, scores, scores_all, boxes_lidar, boxes_lidar_var, head_ids, iouThresh)

    results = {
        "pred_labels": grouped_names,
        "pred_scores": grouped_scores,
        "pred_scores_all": grouped_scores_all,
        "pred_boxes": grouped_boxes,
        "pred_vars": grouped_boxes_var,
        "pred_head_ids": grouped_head_ids
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
    names, scores, scores_all, boxes_lidar, boxes_lidar_var, head_ids = [], [], [], [], [], []
    
    # Handle edge cases
    if len(model_outputs) == 0: # No input
        return names, scores, boxes_lidar, boxes_lidar_var, head_ids
    if type(model_outputs) == dict: # batch size of 1
        names.append(model_outputs['pred_labels'])
        scores.append(model_outputs['pred_scores'])
        scores_all.append(model_outputs['pred_scores_all'])
        boxes_lidar.append(model_outputs['pred_boxes'])
        boxes_lidar_var.append(model_outputs['pred_vars'])
        head_ids.append(model_outputs['pred_head'])
        return names, scores, scores_all, boxes_lidar, boxes_lidar_var, head_ids

    for model_output in model_outputs:
        names.append(model_output['pred_labels'])
        scores.append(model_output['pred_scores'])
        scores_all.append(model_output['pred_scores_all'])
        boxes_lidar.append(model_output['pred_boxes'])
        boxes_lidar_var.append(model_output['pred_vars'])
        head_ids.append(model_output['pred_head'])
    return names, scores, scores_all, boxes_lidar, boxes_lidar_var, head_ids

def grouping(names, scores, scores_all, boxes, boxes_var, head_ids, iouThresh):
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
    batch_size = len(boxes)
    grouped_names, grouped_scores, grouped_scores_all, grouped_boxes, \
        grouped_boxes_var, grouped_head_ids = [], [], [], [], [], []

    
    for i in range(batch_size):
        boxes_i_cpu = boxes[i].cpu()
        clusters, _ = grouping_single_sample(boxes_i_cpu[:,[0,1,3,4,6]], iouThresh)
        grouped_names_single_sample, grouped_scores_single_sample, grouped_scores_all_single_sample, \
            grouped_boxes_single_sample, grouped_boxes_var_single_sample, \
                grouped_head_ids_single_sample = [], [], [], [], [], []
        for cluster in clusters:
            grouped_names_single_sample.append(names[i][cluster])
            grouped_scores_single_sample.append(scores[i][cluster])
            grouped_scores_all_single_sample.append(scores_all[i][cluster])
            grouped_boxes_single_sample.append(boxes[i][cluster, :])
            grouped_boxes_var_single_sample.append(boxes_var[i][cluster, :])
            grouped_head_ids_single_sample.append(head_ids[i][cluster])
        grouped_names.append(grouped_names_single_sample)
        grouped_scores.append(grouped_scores_single_sample)
        grouped_scores_all.append(grouped_scores_all_single_sample)
        grouped_boxes.append(grouped_boxes_single_sample)
        grouped_boxes_var.append(grouped_boxes_var_single_sample)
        grouped_head_ids.append(grouped_head_ids_single_sample)

    return grouped_names, grouped_scores, grouped_scores_all, grouped_boxes, grouped_boxes_var, grouped_head_ids

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
