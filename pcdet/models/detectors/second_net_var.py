from .second_net import SECONDNet
from ..model_utils import dataset_utils

class SECONDNetVAR(SECONDNet):
    def post_processing(self, batch_dict):
        return dataset_utils.post_processing_var(self, batch_dict)
