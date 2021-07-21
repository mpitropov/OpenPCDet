import numpy as np

from ...utils import box_utils
from .kitti_dataset import KittiDataset

class KittiDatasetMIMOVARC(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # For MIMO
        self.NUM_HEADS = dataset_cfg.NUM_HEADS

    # Override the internal generate prediction dict function to perform it for each head
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
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
        FRAME_NUM = 0 # Must have eval set to batch size of 1
        ret_dict = self.orig_generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path)

        # Generate prediction dicts for each head
        ret_dict_list = []
        for i in range(self.NUM_HEADS):
            ret_dict_list.append(self.orig_generate_prediction_dicts( \
                batch_dict, pred_dicts[FRAME_NUM]['pred_dicts_list'][i], class_names, output_path)[FRAME_NUM])
        ret_dict[FRAME_NUM]['post_nms_head_outputs'] = ret_dict_list

        return ret_dict

    @staticmethod
    def orig_generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
                target_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples, num_anchors):
            ret_dict = {
                'name': np.zeros(num_samples),
                'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples),
                'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]),
                'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'score_all': np.zeros([num_samples, len(class_names)+1]),
                'boxes_lidar': np.zeros([num_samples, 7]),
                'pred_labels': np.zeros(num_samples),
                'target_labels': np.zeros(num_samples),
                'pred_vars': np.zeros([num_samples, 7]),
                'pred_head_ids': np.zeros(num_samples),
                # 'anchor_scores': np.zeros(num_anchors),
                # 'anchor_boxes': np.zeros([num_anchors, 7]),
                # 'anchor_labels': np.zeros(num_anchors),
                # 'anchor_vars': np.zeros([num_anchors, 7]),
                # 'selected': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_scores_all = box_dict['pred_scores_all'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            if 'target_labels' in box_dict:
                target_labels = box_dict['target_labels'].cpu().numpy()
            pred_vars = box_dict['pred_vars'].cpu().numpy()
            if 'pred_head_ids' in box_dict:
                pred_head_ids = box_dict['pred_head_ids'].cpu().numpy()
            # anchor_scores = box_dict['anchor_scores'].cpu().numpy()
            # anchor_boxes = box_dict['anchor_boxes'].cpu().numpy()
            # anchor_labels = box_dict['anchor_labels'].cpu().numpy()
            # anchor_vars = box_dict['anchor_vars'].cpu().numpy()
            # selected = box_dict['selected'].cpu().numpy()

            pred_dict = get_template_prediction(pred_scores.shape[0], 0)
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['score_all'] = pred_scores_all
            pred_dict['boxes_lidar'] = pred_boxes[:,:7]
            pred_dict['pred_labels'] = pred_labels
            if 'target_labels' in box_dict:
                pred_dict['target_labels'] = target_labels
            pred_dict['pred_vars'] = pred_vars[:,:7]
            if 'pred_head_ids' in box_dict:
                pred_dict['pred_head_ids'] = pred_head_ids
            # pred_dict['anchor_scores'] = anchor_scores
            # pred_dict['anchor_boxes'] = anchor_boxes[:,:7]
            # pred_dict['anchor_labels'] = anchor_labels
            # pred_dict['anchor_vars'] = anchor_vars[:,:7]
            # pred_dict['selected'] = selected

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos
