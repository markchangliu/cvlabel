from typing import TypedDict, Union, Optional, Literal, Any, List

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

from cvstruct.typedef.bboxes import BBoxLabelmeType
from cvstruct.typedef.polys import PolyLabelmeType


class LabelmeShapeDict(TypedDict):
    points: Union[PolyLabelmeType, BBoxLabelmeType]
    label: str
    shape_type: Literal["polygon", "rectangle"]
    group_id: Optional[str]
    flags: Dict[Any, Any]

class LabelmeDict(TypedDict):
    version: str
    flags: Dict[Any, Any]
    shapes: List[LabelmeShapeDict]
    imagePath: str
    imageData: Optional[str]
    imageHeight: int
    imageWidth: int