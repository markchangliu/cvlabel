from typing import TypedDict, Literal, Union, List

try:
    from typing import TypeAlias
except:
    from typing_extensions import TypeAlias

from cvstruct.typedef.bboxes import BBoxCOCOType
from cvstruct.typedef.polys import PolysCocoType
from cvstruct.typedef.masks import RLEsType


COCO_IMG_TEMPLATE = {
    "height": 0,
    "width": 0,
    "id": 0,
    "file_name": ""
}

COCO_CAT_TEMPLATE = {
    "id": 0,
    "name": ""
}

COCO_ANN_TEMPLATE = {
    "id": 0,
    "iscrowd": 0,
    "image_id": 0,
    "area": 0,
    "bbox": [],
    "segmentation": [],
}

COCO_TEMPLATE = {
    "images": COCO_IMG_TEMPLATE,
    "categories": COCO_CAT_TEMPLATE,
    "annotations": COCO_ANN_TEMPLATE
}


EmptyListType: TypeAlias = List[int]


class COCOImgDict(TypedDict):
    height: int
    width: int
    id: int
    file_name: str

class COCOCatDict(TypedDict):
    id: int
    name: str

class COCOAnnDict(TypedDict):
    id: int
    iscrowd: Literal[0, 1]
    image_id: int
    area: int
    bbox: BBoxCOCOType
    segmentation: Union[PolysCocoType, RLEsType, EmptyListType]

class COCODict(TypedDict):
    images: List[COCOImgDict]
    categories: List[COCOCatDict]
    annotations: List[COCOAnnDict]