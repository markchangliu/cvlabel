import json
import os
from typing import Dict, Union

import numpy as np
import pycocotools.mask as pycocomask

from cvstruct.typedef.insts import InstsType


def labelme2insts_bbox(
    labelme_p: Union[os.PathLike, str],
    cat_name_id_dict: Dict[str, int],
) -> InstsType:
    with open(labelme_p, "r") as f:
        labelme_dict = json.load(f)
    
    anns = labelme_dict["shapes"]
    cat_ids = []
    bboxes = []

    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    for ann in anns:
        cat_id = cat_name_id_dict[ann["label"]]
        x1y1, x2y2 = ann["points"]
        x1, y1 = x1y1
        x2, y2 = x2y2
        bbox = [x1, y1, x2, y2]

        bboxes.append(bbox)
        cat_ids.append(cat_id)
    
    cat_ids = np.asarray(cat_ids, dtype=np.int32)
    scores = np.ones_like(cat_ids, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.int32)

    insts = InstsType(scores, cat_ids, bboxes, None)
    
    return insts