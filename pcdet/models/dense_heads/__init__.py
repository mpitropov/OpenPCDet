from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .anchor_head_single_var import AnchorHeadSingleVAR
from .anchor_head_multi_var import AnchorHeadMultiVAR
from .anchor_head_mimo import AnchorHeadMIMO

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'AnchorHeadMultiVAR': AnchorHeadMultiVAR,
    'AnchorHeadSingleVAR': AnchorHeadSingleVAR,
    'AnchorHeadMIMO': AnchorHeadMIMO
}
