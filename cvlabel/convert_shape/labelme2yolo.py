from typing import List, Dict, Tuple

import cvlabel.typedef.labelme as labelme_type
import cvlabel.typedef.yolo as yolo_type
import cvstruct.merge.cnts as cnt_merge
import cvstruct.convert.poly2cnt as poly_cvt
import cvstruct.convert.cnt2poly as cnt_cvt


def shape_groups_to_yolo_poly(
    shape_groups: labelme_type.LabelmeShapeGroupsType,
    cat_name_id_dict: Dict[str, int],
    img_hw: Tuple[int, int]
) -> yolo_type.YoloPolyLabelsType:
    yolo_labels = []

    # Merge shapes with the same group_id
    for group_id, shape_group in shape_groups.items():
        if len(shape_group) == 0:
            print("Empty shape group")
            continue
        if len(shape_group) == 1:
            # Non-occluded instance, 1 poly
            poly_labelme = shape_group[0]["points"]
            cnt = poly_cvt.poly2cnt_labelme(poly_labelme)
        else:
            # Occluded instance, 
            # 1 bbox + multiple polys, or multiple polys
            cnts = []

            for shape in shape_group:
                if shape["shape_type"] == "rectangle":
                    continue

                poly_labelme = shape["points"]
                cnt = poly_cvt.poly2cnt_labelme(poly_labelme)
            
            if len(cnts) == 0:
                continue
            
            cnt = cnt_merge.merge_contours_sibling(cnts)

        if len(cnt) == 0:
            print("Empty contour")
            continue

        poly_yolo = cnt_cvt.cnt2poly_yolo(cnt, img_hw)

        cat_name = shape_group[0]["label"]
        cat_id = cat_name_id_dict[cat_name]

        yolo_label = [cat_id] + poly_yolo
        yolo_labels.append(yolo_label)

    return yolo_labels