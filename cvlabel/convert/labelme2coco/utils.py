import os
from pathlib import Path
from typing import Union, Generator, Dict, Tuple, Any

import pycocotools.mask as pycocomask

import cvlabel.typedef.labelme as labelme_type
from cvstruct.polys.convert import poly_labelme_to_rle


def img_labelme_p_generator(
    img_dir: Union[str, os.PathLike],
    labelme_dir: Union[str, os.PathLike]
) -> Generator[Tuple[str, str], None, None]:
    filenames = os.listdir(img_dir)
    filenames.sort()

    for filename in filenames:
        if not filename.endswith((".png", ".jpeg", ".jpg")):
            continue

        img_name = filename
        img_suffix = Path(img_name).suffix
        img_p = os.path.join(img_dir, img_name)

        labelme_name = img_name.replace(img_suffix, ".json")
        labelme_p = os.path.join(labelme_dir, labelme_name)

        if not os.path.exists(labelme_p):
            continue

        yield img_p, labelme_p

def get_shapes_by_group_id(
    labelme_dict: Dict[str, Any]
) -> labelme_type.LabelmeShapeGroupDictType:
    # Ann buffer to process shapes with same group_id
    shape_group: labelme_type.LabelmeShapeGroupDictType = {}
    group_id_single = 0

    # Put shapes with the same group_id into one list
    for shape in labelme_dict["shapes"]:
        # skip this shape if its cat not in include_cat_names
        shape_cat_name = shape["label"]
        group_id = shape["group_id"]

        if group_id is None:
            group_id = f"null{group_id_single}"
            group_id_single += 1

        if group_id not in shape_group.keys():
            shape_group[group_id] = []

        shape_group[group_id].append(shape)
    
    return shape_group

def merge_shapes_within_group(
    shape_group: labelme_type.LabelmeShapeGroupDictType,
    img_hw: Tuple[int, int]
) -> labelme_type.LabelmeShapeDictType:
    shape_group_merged = {}

    # Merge shapes with the same group_id
    for group_id, group_shapes in shape_group.items():
        if len(group_shapes) == 1:
            # Non-occluded instance, 1 poly
            ann = group_shapes[0]
            poly = ann["points"]

            if len(poly) < 3:
                # 2 points polygon is problematic
                continue

            rle = poly_labelme_to_rle(poly, img_hw)
        else:
            # Occluded instance, 
            # 1 bbox + multiple polys, or multiple polys
            rles = []

            for ann in group_shapes:
                if ann["shape_type"] == "rectangle":
                    continue

                poly = ann["points"]

                if len(poly) < 3:
                    continue

                rle = poly_labelme_to_rle(poly, img_hw)
                rles.append(rle)
            
            if len(rle) == 0:
                continue

            rle = pycocomask.merge(rles, intersect = False)
        
        # rle byte str to str
        rle["counts"] = rle["counts"].decode("utf-8")

        ann_merged = {
            "cat_name": group_shapes[0]["label"],
            "rle": rle
        }

        shape_group_merged[group_id] = ann_merged

    return shape_group_merged