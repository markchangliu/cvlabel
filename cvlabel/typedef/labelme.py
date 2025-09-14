from typing import TypedDict, Union, Optional, Literal, Any, List, Dict

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

from cvstruct.typedef.bboxes import BBoxLabelmeType
from cvstruct.typedef.polys import PolyLabelmeType


class LabelmeShapeDictType(TypedDict):
    """
    `LabelmeShapeDictType`, `dict`
        `points`: `Union[PolyLabelmeType, BBoxLabelmeType]`
        `label`: `str`
        `shape_type`: `Literal["polygon", "rectangle"]`
        `group_id`: `Optional[str]`
        `flags`: `Dict[Any, Any]`
    """
    points: Union[PolyLabelmeType, BBoxLabelmeType]
    label: str
    shape_type: Literal["polygon", "rectangle"]
    group_id: Optional[str]
    flags: Dict[Any, Any]

class LabelmeShapeGroupDictType(TypedDict):
    """
    `LabelmeShapeGroupDictType`, `dict`
        `group_id`: `Union[int, str]`
        `shapes`: `List[LabelmeShapeDict]`
    """
    group_id: Union[int, str]
    shapes: List[LabelmeShapeDict]

class LabelmeDictType(TypedDict):
    """
    `LabelmeDictType`, `dict`
        `version`: `str`
        `flags`: `Dict[Any, Any]`
        `shapes`: `List[LabelmeShapeDict]`
        `imagePath`: `str`
        `imageData`: `Optional[str]`
        `imageHeight`: int
        `imageWidth`: `int`
    """
    version: str
    flags: Dict[Any, Any]
    shapes: List[LabelmeShapeDict]
    imagePath: str
    imageData: Optional[str]
    imageHeight: int
    imageWidth: int