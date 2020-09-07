from .nuscenes_dataset import NuScenesDataset

class NuScenesDatasetVAR(NuScenesDataset):
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples, num_anchors):
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 
                'pred_labels': np.zeros(num_samples),
                'pred_vars': np.zeros([num_samples, 7]),
                'anchor_scores': np.zeros(num_anchors),
                'anchor_boxes': np.zeros([num_anchors, 7]),
                'anchor_labels': np.zeros(num_anchors),
                'anchor_vars': np.zeros([num_anchors, 7]),
                'selected': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_vars = box_dict['pred_vars'].cpu().numpy()
            anchor_scores = box_dict['anchor_scores'].cpu().numpy()
            anchor_boxes = box_dict['anchor_boxes'].cpu().numpy()
            anchor_labels = box_dict['anchor_labels'].cpu().numpy()
            anchor_vars = box_dict['anchor_vars'].cpu().numpy()
            selected = box_dict['selected'].cpu().numpy()

            pred_dict = get_template_prediction(pred_scores.shape[0], anchor_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes[:,:7]
            pred_dict['pred_labels'] = pred_labels
            pred_dict['pred_vars'] = pred_vars[:,:7]
            pred_dict['anchor_scores'] = anchor_scores
            pred_dict['anchor_boxes'] = anchor_boxes[:,:7]
            pred_dict['anchor_labels'] = anchor_labels
            pred_dict['anchor_vars'] = anchor_vars[:,:7]
            pred_dict['selected'] = selected

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

