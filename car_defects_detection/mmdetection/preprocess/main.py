import os
import sys
import json
import time
import random
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-GUI backend for servers
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from mmdet.apis import DetInferencer
import yaml
from configs.config import params
from utils.roi_cropper import ROICropper
from utils.annotations_vlaidation import extract_valid_category_ids, make_file_logger, preprocess_annotations, update_annotation_after_roi_crop

# ------------------------------------------------------------
# Utility to fetch model config
# ------------------------------------------------------------
def get_model_config(model_key: str):
    if model_key not in params.model_key:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {list(params.model_zoo.keys())}"
        )
    cfg = params.model_zoo[model_key]
    return cfg["model_name"], cfg["checkpoint"]

def ann_to_mask(seg, height, width):
    if isinstance(seg, list):  # polygon
        rles = mask_utils.frPyObjects(seg, height, width)
        m = mask_utils.decode(rles)
        return m.max(axis=2)
    else:  # RLE dict
        return mask_utils.decode(seg)
    
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':

    # Validate path
    if not os.path.exists(params.data_dir):
        print("Error: dataset path doesn't exist.")
        sys.exit(1)

    ann_dir = os.path.join(params.data_dir, "annotations")
    img_dir = os.path.join(params.data_dir, f"{params.dataset}2017")

    if not os.path.exists(ann_dir) or not os.path.exists(img_dir):
        print("Error: missing annotation or image folder.")
        sys.exit(1)

    # Ensure output root exists
    os.makedirs(params.output_root, exist_ok=True)
    # ============================================================
    #                   ANN PREPROCESSING
    # ============================================================
   
    pre_ann_path = os.path.join(ann_dir, f"annotations_{params.dataset}.json")

    with open(pre_ann_path, "r") as f:
        coco = json.load(f)

    print(f"\nDataset: {params.dataset}")
    print(f"Input annotation file: {pre_ann_path}\n")
    
    valid_ids = extract_valid_category_ids(coco)
    logger = make_file_logger(f"../logs/preprocess_annotations_{params.dataset}.log")

    cleaned_anns, stats = preprocess_annotations(coco, valid_ids, logger=logger)

    # Build new COCO file
    preprocess_coco = {
        "licenses": coco["licenses"],
        "info": coco["info"],
        "images": coco["images"],
        "annotations": cleaned_anns,
        "categories": coco["categories"],
    }
    out_pre_ann_filename = f"annotations_{params.dataset}_preprocess.json"
    out_pre_ann_fullpath = os.path.join(ann_dir, out_pre_ann_filename)

    with open(out_pre_ann_fullpath, "w") as f:
        json.dump(preprocess_coco, f, indent=2)

    print(f"Saved updated COCO annotations to {out_pre_ann_fullpath}")

    print("\nSUMMARY STATS:")
    for k, v in stats.items():
        print(f"{k:30s}: {v}")

    # ============================================================
    #                   ROI Extraction
    # ============================================================
    postprocess_coco = preprocess_coco.copy()
    # New containers for postprocessed annotations and images
    post_images = []
    post_annotations = []

    # Build fast index mappings for lookups
    image_dict = {img["id"]: img for img in postprocess_coco["images"]}
    ann_list = postprocess_coco["annotations"]
    cat_id_to_name = {c["id"]: c["name"] for c in postprocess_coco["categories"]}

    # Choose subset of image IDs
    img_ids = [img["id"] for img in postprocess_coco["images"]]
    subset_ids = (
        random.sample(img_ids, params.num_images)
        if params.num_images > 0
        else img_ids
    )
    print(f"Selected images: {'all' if params.num_images == 0 else len(subset_ids)}")

    # ============================================================
    #                    LOAD MODEL + CROP LOGIC
    # ============================================================
    model_name, checkpoint = get_model_config(params.model_key)
    print(f"Using model: {params.model_key}")

    inferencer = DetInferencer(model_name, checkpoint)
    class_names = inferencer.model.dataset_meta["classes"]
    cropper = ROICropper(class_names)

    times = []

    # ============================================================
    #                        MAIN LOOP
    # ============================================================
    for img_id in subset_ids:

        # ------------------ Load image --------------------------
        img_info = image_dict[img_id]
        img_path = os.path.join(img_dir, img_info["file_name"])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if image is None:
            print(f"Warning: failed to load {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        # ------------------ Output folder ------------------------
        out_dir = os.path.join(params.output_root, str(img_id))
        os.makedirs(out_dir, exist_ok=True)

        # ------------------ Load anns -----------------------------
        anns = [ann for ann in ann_list if ann["image_id"] == img_id]

        # Optional debug mask visualization
        if params.save_debug_vis:
            vis_img = image.copy()

            for ann in anns:
                color = tuple(np.random.randint(0, 255, 3).tolist())

                # blend segmentation mask
                if ann.get("segmentation"):
                    mask = ann_to_mask(ann["segmentation"], img_h, img_w)
                    vis_img[mask == 1] = (
                        vis_img[mask == 1] * 0.5 + np.array(color) * 0.5
                    )

                # draw bbox
                x, y, w, h = ann["bbox"]
                cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                # class name
                cat_name = cat_id_to_name[ann["category_id"]]
                cv2.putText(
                    vis_img,
                    cat_name,
                    (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            plt.imsave(os.path.join(out_dir, "image_manipulated.jpg"), vis_img)

        # ============================================================
        #                       DETECTOR RUN
        # ============================================================
        torch.cuda.synchronize()
        start = time.time()

        result = inferencer(image, return_vis=True)

        torch.cuda.synchronize()
        times.append(time.time() - start)

        vis_pred = result["visualization"][0]
        pred = result["predictions"][0]

        # Save visualization
        if params.save_debug_vis:
            plt.imsave(
                os.path.join(out_dir, f"detector_vis_{params.model_key}.jpg"),
                vis_pred
            )

        bboxes = np.array(pred["bboxes"])
        scores = np.array(pred["scores"])
        labels = np.array(pred["labels"])

        # ============================================================
        #                   ROI EXTRACTION + UPDATE JSON
        # ============================================================
        roi_info = cropper.extract_roi(
            bboxes,
            scores,
            labels,
            img_w,
            img_h,
            img_id,
            mode="training",
        )

        if roi_info is None:
            continue  # skip ROI for this image

        roi, cls_id, cls_name, score = roi_info

        # convert ROI to COCO [x,y,w,h]
        x1, y1, x2, y2 = map(int, roi)
        roi_coco = [x1, y1, x2 - x1, y2 - y1]
        
        # Save updated image entry
        img_entry = image_dict[img_id]
        img_entry["roi_bbox"] = roi_coco
        post_images.append(img_entry)

        
        for ann in anns:
            # new_ann = ann.copy()
            # # Update bbox
            # new_ann["bbox"] =
            # if new_ann["bbox"] is None:
            #     continue
            # # Crop segmentation mask
            # new_ann["segmentation"] = 
            # # Update area
            # new_ann["area"] =
            # post_annotations.append(new_ann)
            ok = update_annotation_after_roi_crop(
                    ann,
                    roi,
                    img_h,
                    img_w
                )
            if ok:
                post_annotations.append(ann)

        # save cropped ROI image
        if params.save_debug_vis:
            roi_img = image[y1:y2, x1:x2]
            plt.imsave(os.path.join(out_dir, f"cropped_roi_{params.model_key}.jpg"), roi_img)

    # Update postprocess_coco
    postprocess_coco["annotations"] = post_annotations
    postprocess_coco["images"] = post_images

    
    # ============================================================
    #                SAVE UPDATED JSON WITH ROI
    # ============================================================
    out_post_ann_filename = f"annotations_{params.dataset}_postprocess.json"
    out_post_ann_fullpath = os.path.join(ann_dir, out_post_ann_filename)

    with open(out_post_ann_fullpath, "w") as f:
        json.dump(postprocess_coco, f, indent=2)

    print("\nSaved updated COCO annotations with ROI to")
    print(out_post_ann_fullpath)

    # ============================================================
    #                    SHOW ROI CROP STATS
    # ============================================================
    stats = cropper.get_statistics()
    cropper.show_hist(os.path.join(params.output_root, f"hist_th_{params.score_threshold}_{params.model_key}.jpg"),
                      params.save_debug_vis)

    print(f"\nSUMMARY STATS  (model={params.model_key}, th={params.score_threshold}):")
    for k, v in stats.items():
        print(f"{k:30s}: {v}")

    print(f"\nAverage inference time: {np.mean(times):.3f} sec over {len(subset_ids)} images")
    print("\nDone.")

    # ------------------------------------
    # Save stats to YAML file
    # ------------------------------------
    stats_path = os.path.join(params.output_root, f"summary_stats_{params.model_key}_th{params.score_threshold}.yml")
    
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f, sort_keys=False, default_flow_style=False)
    
    print(f"\nSaved summary stats to: {stats_path}")
