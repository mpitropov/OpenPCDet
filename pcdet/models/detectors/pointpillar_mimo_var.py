import torch

from .pointpillar import PointPillar
from ..model_utils import model_nms_utils
from scipy.stats import circmean

class PointPillarMIMOVAR(PointPillar):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.NUM_HEADS = model_cfg.DENSE_HEAD.NUM_HEADS
        self.OUTPUT_PRED_LIST = model_cfg.OUTPUT_PRED_LIST

    def post_processing(self, batch_dict):
        """
        For MIMO call post processing on each head then use IoU to cluster
        """

        mimo_batch_size = int(batch_dict['batch_size'] / self.NUM_HEADS)

        if mimo_batch_size != 1:
            print("Testing MIMO can only be set to batch of 1")
            raise NotImplementedError

        batch_dict_list = []
        pred_dicts_list = []
        recall_dict_list = []
        for i in range(self.NUM_HEADS):
            if i == 0:
                batch_dict_list.append({
                    'batch_size': mimo_batch_size,
                    'batch_features': batch_dict['batch_features'],
                    'batch_cls_preds': batch_dict['batch_cls_preds'],
                    'batch_box_preds': batch_dict['batch_box_preds'],
                    'batch_var_preds': batch_dict['batch_var_preds'],
                    'cls_preds_normalized': batch_dict['cls_preds_normalized']
                })
            else:
                batch_dict_list.append({
                    'batch_size': mimo_batch_size,
                    'batch_features': batch_dict['batch_features_' + str(i)],
                    'batch_cls_preds': batch_dict['batch_cls_preds_' + str(i)],
                    'batch_box_preds': batch_dict['batch_box_preds_' + str(i)],
                    'batch_var_preds': batch_dict['batch_var_preds_' + str(i)],
                    'cls_preds_normalized': batch_dict['cls_preds_normalized_' + str(i)]
                })

            tmp_pred_dicts, tmp_recall_dict = self.post_processing_single(batch_dict_list[i])
            pred_dicts_list.append(tmp_pred_dicts)
            recall_dict_list.append(tmp_recall_dict)

        if self.OUTPUT_PRED_LIST:
            # Create dict output with one car so that OpenPCDet does not error
            pred_dicts = [{
                'feature': torch.zeros(1),
                'pred_labels': torch.zeros(1, dtype=torch.int32).fill_(1), # Car
                'pred_scores': torch.zeros(1).fill_(0.0001), # Low prediction
                'pred_scores_all': torch.zeros(1,4).fill_(0.0001),
                'pred_boxes': torch.zeros(1,7),
                'pred_vars': torch.zeros(1,7),
                'pred_head_ids': torch.zeros(11).fill_(0), # mean head
                'pred_dicts_list': pred_dicts_list
            }]
            return pred_dicts, recall_dict_list[0]


        ONLY_HEAD_0 = False
        if ONLY_HEAD_0:
            print('Only using output from head 0')
            # Set all the detection head ids to 0, the "mean" head
            pred_dicts_list[0][0]['pred_head_ids'] = torch.zeros(len(pred_dicts_list[0][0]['pred_labels']))
            return pred_dicts_list[0], recall_dict_list[0]

        # Instead of doing the concat we test having an extra NMS
        EXTRA_NMS_TEST = False
        if EXTRA_NMS_TEST == True:
            print('Performing extra NMS')
            batch_cls_preds = torch.cat( \
                [tmp_pred_dicts[0]['pred_scores_all'] for tmp_pred_dicts in pred_dicts_list] \
                , 0) # (num_boxes, num_classes)
            batch_cls_preds = torch.unsqueeze(batch_cls_preds, 0) # (N, num_boxes, num_classes)
            batch_box_preds = torch.cat( \
                [tmp_pred_dicts[0]['pred_boxes'] for tmp_pred_dicts in pred_dicts_list] \
                , 0) # (num_boxes, 7)
            batch_box_preds = torch.unsqueeze(batch_box_preds, 0) # (N, num_boxes, 7)
            batch_var_preds = torch.cat( \
                [tmp_pred_dicts[0]['pred_vars'] for tmp_pred_dicts in pred_dicts_list] \
                , 0) # (num_boxes, 7)
            batch_var_preds = torch.unsqueeze(batch_var_preds, 0) # (N, num_boxes, 7)
            batch_dict_extra_nms = {
                'batch_size': mimo_batch_size,
                'batch_features': batch_dict['batch_features'], # from head A
                'batch_cls_preds': batch_cls_preds,
                'batch_box_preds': batch_box_preds,
                'batch_var_preds': batch_var_preds,
                # We can't add cls_preds_normalized since NMS doesn't return it
                'cls_preds_normalized': None
            }
            pred_dicts, recall_dict = self.post_processing_single(batch_dict_extra_nms)
            # Set all the detection head ids to 0, the "mean" head
            pred_dicts[0]['pred_head_ids'] = torch.zeros(len(pred_dicts[0]['pred_labels']))
            return pred_dicts, recall_dict

        # Array of detections dictionaries to return
        pred_dicts = []

        # Perform clustering for each item in batch size
        for batch_index in range(mimo_batch_size):
            # batch_size, prediction dictionary, list
            num_preds_per_head = []
            head_id_list = []
            for i in range(self.NUM_HEADS):
                num_preds_per_head.append(len(pred_dicts_list[i][batch_index]['pred_labels']))
                head_id_list.append(torch.empty(num_preds_per_head[i]).fill_(i+1))

            iou_input = [{
                'pred_labels': torch.cat( \
                    [tmp_pred_dicts[batch_index]['pred_labels'] \
                        for tmp_pred_dicts in pred_dicts_list], 0),
                'pred_scores': torch.cat( \
                    [tmp_pred_dicts[batch_index]['pred_scores'] \
                        for tmp_pred_dicts in pred_dicts_list], 0),
                'pred_scores_all': torch.cat( \
                    [tmp_pred_dicts[batch_index]['pred_scores_all'] \
                        for tmp_pred_dicts in pred_dicts_list], 0),
                'pred_boxes': torch.cat( \
                    [tmp_pred_dicts[batch_index]['pred_boxes'] \
                        for tmp_pred_dicts in pred_dicts_list], 0),
                'pred_vars': torch.cat( \
                    [tmp_pred_dicts[batch_index]['pred_vars'] \
                        for tmp_pred_dicts in pred_dicts_list], 0),
                'pred_head': torch.cat(head_id_list, 0),
            }]

            cluster_results = preprocess(iou_input, iouThresh=0.7)
            total_num_clusters = len(cluster_results['pred_labels'][0])

            # temporary storage
            # NOTE: We ignore 'anchor' outputs and 'selected' output
            pred_labels = []
            pred_scores = []
            pred_scores_all = []
            pred_boxes = []
            pred_vars = []
            pred_head_ids = []

            # Merge cluster output into a single prediction per cluster
            MIN_PREDS_IN_CLUSTER = self.NUM_HEADS # - 1 # Only one bad head allowed
            NUM_BAD = 0
            NUM_GOOD = 0
            OUTPUT_ALL = False
            # if OUTPUT_ALL:
            #     print('Debug outputting mean detection + all head detections')
            AVG_PRED_HEAD_ID = torch.tensor(0)
            for cluster_num in range(total_num_clusters):
                num_preds_in_cluster = len(cluster_results['pred_labels'][0][cluster_num])

                # This will output all prediction head outputs for visualization
                if OUTPUT_ALL:
                    for pred_idx in range(num_preds_in_cluster):
                        pred_labels.append(cluster_results['pred_labels'][0][cluster_num][pred_idx])
                        pred_scores.append(cluster_results['pred_scores'][0][cluster_num][pred_idx])
                        pred_scores_all.append(cluster_results['pred_scores_all'][0][cluster_num][pred_idx])
                        pred_boxes.append(cluster_results['pred_boxes'][0][cluster_num][pred_idx])
                        pred_vars.append(cluster_results['pred_vars'][0][cluster_num][pred_idx])
                        pred_head_ids.append(cluster_results['pred_head_ids'][0][cluster_num][pred_idx])

                # Skip when heads are not certain
                if num_preds_in_cluster < MIN_PREDS_IN_CLUSTER:
                    NUM_BAD += 1
                    continue
                # Check if there are more detections in cluster than prediction heads
                if num_preds_in_cluster > self.NUM_HEADS:
                    print("number of predictions in cluster more than", self.NUM_HEADS)
                    print(cluster_results['pred_head_ids'][0][cluster_num])
                    exit()
                # Check if there is output from the same head in this cluster
                hist = torch.histc(cluster_results['pred_head_ids'][0][cluster_num], \
                                    bins=3, min=1, max=3)
                if torch.max(hist) > 1:
                    print("A detection head has contributed more than once to this cluster")
                    print('bin histogram', hist)
                    exit()
                # Check if predictions in cluster have different labels
                hist = torch.histc(cluster_results['pred_labels'][0][cluster_num], \
                                    bins=4, min=0, max=3)
                if torch.max(hist) != num_preds_in_cluster:
                    # If there are two preds in cluster, then this is not a min cluster
                    if num_preds_in_cluster == 2 or num_preds_in_cluster == 3:
                        continue
                    print("This cluster has multiple predicted labels")
                    print('bin histogram', hist)
                    print('Pred labels', cluster_results['pred_labels'][0][cluster_num])
                    print('Top Score', cluster_results['pred_scores'][0][cluster_num])
                    print('SoftMax Score output', cluster_results['pred_scores_all'][0][cluster_num])
                    exit()

                # The cluster is valid
                NUM_GOOD += 1
                # Now create the average head output
                # This does not need to take the mean just take the first one
                pred_labels.append(cluster_results['pred_labels'][0][cluster_num][0])
                # Take the means otherwise
                pred_scores.append(torch.mean(cluster_results['pred_scores'][0][cluster_num]))
                pred_scores_all.append(torch.mean(cluster_results['pred_scores_all'][0][cluster_num], dim=0))
                pred_box_tmp = torch.mean(cluster_results['pred_boxes'][0][cluster_num], dim=0)
                # # We must use circmean for edge case angles
                # angles = cluster_results['pred_boxes'][0][cluster_num][:,6]
                # circmean_of_angles = circmean(angles.cpu(), high = np.pi, low = -np.pi)
                # testing = circmean([pred_box_tmp[6].cpu()], high = np.pi, low = -np.pi)
                # if np.abs(testing - circmean_of_angles) > 1.0:
                #     print("circ mean test")
                #     print("pred_boxes", cluster_results['pred_boxes'][0][cluster_num])
                #     print("pred_boxes mean", torch.mean(cluster_results['pred_boxes'][0][cluster_num], dim=0))
                #     print("angles to take mean of", angles)
                #     print("real mean of angles", circmean_of_angles)
                #     print("fake mean of angles", pred_box_tmp[6])
                #     exit()
                # pred_box_tmp[6] = circmean_of_angles

                # Use the orientation of the box with highest confidence
                # Using circmean causes problems with orientations flipped 90 deg
                highest_conf_pred_idx = torch.argmax(cluster_results['pred_scores'][0][cluster_num])
                pred_box_tmp[6] = cluster_results['pred_boxes'][0][cluster_num][highest_conf_pred_idx][6]
                # print(cluster_results['pred_scores'][0][cluster_num])
                # print(highest_conf_pred_idx)
                # print(cluster_results['pred_boxes'][0][cluster_num])
                # print(pred_box_tmp[6])
                # exit()
                pred_boxes.append(pred_box_tmp)
                pred_vars.append(torch.mean(cluster_results['pred_vars'][0][cluster_num], dim=0))

                # We have the average head as id 0 and the other heads are 1,2,3
                pred_head_ids.append(AVG_PRED_HEAD_ID)

            # print('NUM_BAD', NUM_BAD)
            # print('NUM_GOOD', NUM_GOOD)
            # print(pred_labels)
            # print(len(pred_labels))

            # Make array of dicts with lists
            # Also stack individual tensors
            if len(pred_labels) > 0:
                pred_dicts.append({
                    'feature': batch_dict['batch_features'], # NOTE: features of only head A
                    'pred_labels': torch.stack(pred_labels),
                    'pred_scores': torch.stack(pred_scores),
                    'pred_scores_all': torch.stack(pred_scores_all),
                    'pred_boxes': torch.stack(pred_boxes),
                    'pred_vars': torch.stack(pred_vars),
                    'pred_head_ids': torch.stack(pred_head_ids)
                })
            else:
                # Send over one fake detection if there are no valid detections
                pred_dicts.append({
                    'feature': batch_dict['batch_features'], # NOTE: features of only head A
                    'pred_labels': torch.zeros(1, dtype=torch.int32).fill_(1), # Car
                    'pred_scores': torch.zeros(1).fill_(0.0001), # Low prediction
                    'pred_scores_all': torch.zeros(1,4).fill_(0.0001),
                    'pred_boxes': torch.zeros(1,7),
                    'pred_vars': torch.zeros(1,7),
                    'pred_head_ids': torch.zeros(11).fill_(0) # mean head
                })

        # Add individual head outputs as extra dict
        pred_dicts[0]['pred_dicts_list'] = []
        for i in range(len(pred_dicts_list)):
            pred_dicts[0]['pred_dicts_list'].append(pred_dicts_list[i])

        return pred_dicts, recall_dict_list[0]

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
            var_preds = batch_dict['batch_var_preds'][batch_mask]
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
