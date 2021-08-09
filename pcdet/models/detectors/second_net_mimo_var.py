from .second_net import SECONDNet
from ..model_utils import dataset_utils

class SECONDNetMIMOVAR(SECONDNet):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.NUM_HEADS = model_cfg.DENSE_HEAD.NUM_HEADS
        self.OUTPUT_PRED_LIST = model_cfg.OUTPUT_PRED_LIST

    def post_processing(self, batch_dict):
        """
        For MIMO call post processing var on each head
        If you are using OpenPCDet to evaluate then it will use IoU to cluster
        """
        return dataset_utils.post_processing_mimo(self, batch_dict)
