from .pointpillar import PointPillar
from ..model_utils import dataset_utils

class PointPillarVAR(PointPillar):
    def post_processing(self, batch_dict):
        return dataset_utils.post_processing_var(self, batch_dict)
