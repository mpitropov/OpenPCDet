from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_var import SECONDNetVAR
from .second_net_mimo_var import SECONDNetMIMOVAR
from .pointpillar_var import PointPillarVAR
from .pointpillar_mimo_var import PointPillarMIMOVAR

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'SECONDNetVAR': SECONDNetVAR,
    'SECONDNetMIMOVAR': SECONDNetMIMOVAR,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'PointPillarVAR': PointPillarVAR,
    'PointPillarMIMOVAR': PointPillarMIMOVAR
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
