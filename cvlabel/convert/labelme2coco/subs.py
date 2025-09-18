import copy
import os
import shutil
from pathlib import Path
from typing import Union, Any, Dict, Tuple, List

import numpy as np
import pycocotools.mask as pycocomask

import cvlabel.typedef.coco as coco_type
import cvlabel.typedef.labelme as labelme_type
import cvlabel.utils.labelme as utils


def labelme2coco_sub_img(
    img_p: Union[str, os.PathLike],
    labelme_dict: labelme_type.LabelmeDictType,
    img_id: int,
    abs_img_p_flag: bool,
    img_p_prefix: Union[str, os.PathLike],
) -> coco_type.CocoImgDictType:
    # Add img to img_list
    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    img_name = Path(img_p).name
    img_suffix = Path(img_p).suffix

    file_name = img_p

    if not abs_img_p_flag:
        file_name = str(Path(img_p).relative_to(img_p_prefix))
    
    coco_img = copy.deepcopy(coco_type.COCO_IMG_TEMPLATE)
    coco_img["height"] = img_h
    coco_img["width"] = img_w
    coco_img["id"] = img_id
    coco_img["file_name"] = file_name

    return coco_img

def labelme2coco_sub_img_copy(
    img_p: Union[str, os.PathLike],
    labelme_dict: labelme_type.LabelmeDictType,
    img_id: int,
    export_img_dir: Union[str, os.PathLike],
    abs_img_p_flag: bool,
) -> coco_type.CocoImgDictType:
    # Add img to img_list
    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]

    img_name = Path(img_p).name
    img_suffix = Path(img_p).suffix

    copy_img_name = f"img{img_id}{img_suffix}"
    file_name = os.path.join(export_img_dir, copy_img_name)
    shutil.copy(img_p, file_name)
        
    if not abs_img_p_flag:
        file_name = copy_img_name
    
    coco_img = copy.deepcopy(coco_type.COCO_IMG_TEMPLATE)
    coco_img["height"] = img_h
    coco_img["width"] = img_w
    coco_img["id"] = img_id
    coco_img["file_name"] = file_name

    return coco_img

def labelme2coco_sub_ann_rle(
    labelme_dict: labelme_type.LabelmeDictType,
    start_ann_id: int,
    cat_name_id_dict: Dict[str, int],
    img_id: int
) -> Tuple[coco_type.CocoAnnDictType, int]:
    img_h = labelme_dict["imageHeight"]
    img_w = labelme_dict["imageWidth"]
    img_hw = (img_h, img_w)

    coco_ann_list = []
    ann_id = start_ann_id

    shape_group = utils.get_shapes_by_group_id(labelme_dict)
    shape_group = utils.merge_shapes_within_group(shape_group, img_hw)

    for _, ann in shape_group.items():
        cat_name = ann["cat_name"]

        if cat_name not in cat_name_id_dict.keys():
            continue

        cat_id = cat_name_id_dict[cat_name]
        rle = ann["rle"]
        bbox_xywh = pycocomask.toBbox(rle).astype(np.int_).tolist()
        area = pycocomask.area(rle).item()

        coco_ann = copy.deepcopy(coco_type.COCO_ANN_TEMPLATE)
        coco_ann["iscrowd"] = 0
        coco_ann["category_id"] = cat_id
        coco_ann["image_id"] = img_id
        coco_ann["bbox"] = bbox_xywh
        coco_ann["area"] = area
        coco_ann["id"] = ann_id
        coco_ann["segmentation"] = rle

        coco_ann_list.append(coco_ann)
        ann_id += 1
    
    return coco_ann_list, ann_id

def labelme2coco_sub_cat(
    cat_name_id_dict: Dict[str, int]
) -> List[coco_type.CocoCatDictType]:
    coco_cat_list = []

    for cat_name, cat_id in cat_name_id_dict.items():
        cat_info = copy.deepcopy(coco_type.COCO_CAT_TEMPLATE)
        cat_info["id"] = cat_id
        cat_info["name"] = cat_name
        coco_cat_list.append(cat_info)

    return coco_cat_list