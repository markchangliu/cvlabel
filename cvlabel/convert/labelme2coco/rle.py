import copy
import json
import os
import shutil
from pathlib import Path
from typing import Union, Dict, List, Tuple, Any

import numpy as np
import pycocotools.mask as pycocomask

import cvlabel.typedef.labelme as labelme_type
import cvlabel.convert.labelme2coco.subs as subs
import cvlabel.utils.labelme as labelme_utils


def labelme2coco_rle_copy_img(
    img_dirs: List[Union[str, os.PathLike]],
    labelme_dirs: List[Union[str, os.PathLike]],
    export_root: Union[str, os.PathLike],
    export_coco_name: str,
    export_img_foldername: str,
    abs_img_p_flag: bool,
    cat_name_id_dict: Dict[str, Any]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    export_img_dir = os.path.join(export_root, export_img_foldername)
    export_coco_p = os.path.join(export_root, export_coco_name)

    os.makedirs(export_img_dir, exist_ok = True)

    # Initialize img, category, annotation ids
    img_id = 0
    cat_id = 0
    ann_id = 0

    # Initialize img, category, annotation list
    coco_img_list = []
    coco_cat_list = []
    coco_ann_list = []

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        p_generator = labelme_utils.img_labelme_p_generator(img_dir, labelme_dir)

        for img_p, labelme_p in p_generator:
            with open(labelme_p, "r") as f:
                labelme_dict: labelme_type.LabelmeDict = json.load(f)

            coco_img = subs.labelme2coco_sub_img_copy(
                img_p, labelme_dict, img_id, export_img_dir, abs_img_p_flag
            )
            coco_ann_list_curr_img, end_ann_id = subs.labelme2coco_sub_ann_rle(
                labelme_dict, ann_id, cat_name_id_dict, img_id
            )

            coco_img_list.append(coco_img)
            coco_ann_list += coco_ann_list_curr_img

            img_id += 1
            ann_id = end_ann_id
    
    coco_cat_list = subs.labelme2coco_sub_cat(cat_name_id_dict)

    coco_ann = {
        "images": coco_img_list,
        "categories": coco_cat_list,
        "annotations": coco_ann_list
    }

    with open(export_coco_p, "w") as f:
        json.dump(coco_ann, f)

def labelme2coco_rle(
    img_dirs: List[Union[str, os.PathLike]],
    labelme_dirs: List[Union[str, os.PathLike]],
    export_coco_p: Union[str, os.PathLike],
    abs_img_p_flag: bool,
    img_p_prefix: Union[str, os.PathLike],
    cat_name_id_dict: Dict[str, Any]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    export_coco_dir = Path(export_coco_p).parent
    os.makedirs(export_coco_dir, exist_ok = True)

    # Initialize img, category, annotation ids
    img_id = 0
    cat_id = 0
    ann_id = 0

    # Initialize img, category, annotation list
    coco_img_list = []
    coco_cat_list = []
    coco_ann_list = []

    for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
        p_generator = labelme_utils.img_labelme_p_generator(img_dir, labelme_dir)

        for img_p, labelme_p in p_generator:
            with open(labelme_p, "r") as f:
                labelme_dict = json.load(f)

            coco_img = subs.labelme2coco_sub_img(
                img_p, labelme_dict, img_id, abs_img_p_flag, img_p_prefix,
            )
            coco_ann_list_curr_img, end_ann_id = subs.labelme2coco_sub_ann_rle(
                labelme_dict, ann_id, cat_name_id_dict, img_id
            )

            coco_img_list.append(coco_img)
            coco_ann_list += coco_ann_list_curr_img

            img_id += 1
            ann_id = end_ann_id
    
    coco_cat_list = subs.labelme2coco_sub_cat(cat_name_id_dict)

    coco_ann = {
        "images": coco_img_list,
        "categories": coco_cat_list,
        "annotations": coco_ann_list
    }

    with open(export_coco_p, "w") as f:
        json.dump(coco_ann, f)