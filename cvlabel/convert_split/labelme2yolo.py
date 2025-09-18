import os
from typing import Union, List, Dict

import cvlabel.convert_batch.labelme2yolo as cvt_batch


def labelme2coco_poly_split(
    img_dirs: Union[str, os.PathLike],
    labelme_dirs: Union[str, os.PathLike],
    export_root: Union[str, os.PathLike],
    export_foldernames: List[str],
    splits_kwords: List[List[str]],
    cat_name_id_dict: Dict[str, int]
) -> None:
    assert isinstance(img_dirs, list)
    assert isinstance(labelme_dirs, list)
    assert len(img_dirs) == len(labelme_dirs)

    splits_convert_kwargs = []

    for kws, foldername in zip(splits_kwords, export_foldernames):
        split_convert_kwargs = {
            "img_dirs": [],
            "labelme_dirs": [],
            "export_root": export_root,
            "export_foldername": foldername,
            "cat_name_id_dict": cat_name_id_dict
        }

        for img_dir, labelme_dir in zip(img_dirs, labelme_dirs):
            include_flags = []
            
            for kw in kws:
                if kw not in img_dir:
                    include_flags.append(False)
                    break
                else:
                    include_flags.append(True)
                    continue
            
            if not all(include_flags):
                continue

            split_convert_kwargs["img_dirs"].append(img_dir)
            split_convert_kwargs["labelme_dirs"].append(labelme_dir)

            splits_convert_kwargs.append(split_convert_kwargs)
    
    for kwargs in splits_convert_kwargs:
        cvt_batch.labelme2yolo_poly(**kwargs)
        
