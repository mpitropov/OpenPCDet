import torch

from ..model_utils import model_nms_utils
from ..model_utils import cluster_utils
from scipy.stats import circmean

# post_processing function with predicted variance
def post_processing_var(self, batch_dict):
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

        # cls_targets = batch_dict['batch_cls_targets'][batch_mask]
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
            # final_target_labels = cls_targets
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
            # 'target_labels': final_target_labels[final_selected],
            'pred_vars': final_vars[final_selected],
            'anchor_boxes': final_boxes,
            'anchor_scores': final_scores,
            'anchor_labels': final_labels,
            'anchor_vars': final_vars,
            'selected': final_selected
        }
        pred_dicts.append(record_dict)

    return pred_dicts, recall_dict

def post_processing_mimo(self, batch_dict):
    """
    For MIMO call post processing var on each head
    If you are using OpenPCDet to evaluate then it will use IoU to cluster
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

        tmp_pred_dicts, tmp_recall_dict = post_processing_var(self, batch_dict_list[i])
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
        pred_dicts, recall_dict = post_processing_var(self, batch_dict_extra_nms)
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

        cluster_results = cluster_utils.preprocess(iou_input, iouThresh=0.7)
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
