import json
import os
from datetime import datetime
from collections import defaultdict
from pycocotools import mask as mask_utils
import numpy as np


# =====================================================================
# Logging utilities
# =====================================================================

def make_file_logger(log_path):
    """
    Create a simple file logger(msg) that appends timestamped lines
    to the given log file path.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def logger(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")

    return logger


# =====================================================================
# Phase 1 — Geometry corrections (bbox + polygon)
# =====================================================================

def clip_box_to_image(box, W, H):
    """
    Clip COCO bbox [x, y, w, h] to image boundaries [0..W-1], [0..H-1].
    Returns:
        new_box or None if invalid after clipping.
    """
    if len(box) != 4:
        return None

    x, y, w, h = box
    x2, y2 = x + w, y + h

    # Fix reversed width or height (if annotator accidentally swapped)
    if w < 0:
        x, x2 = x2, x
    if h < 0:
        y, y2 = y2, y

    # Clip coordinates
    x  = max(0, min(x,  W - 1))
    y  = max(0, min(y,  H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))

    if x2 <= x or y2 <= y:
        return None

    return [float(x), float(y), float(x2 - x), float(y2 - y)]

def clip_polygon(poly, xend, yend, xstart=0, ystart=0):
    """
    Clip polygon segmentation list [x1, y1, x2, y2, ...] to image bounds.
    Minimum required length is 6 (3 points).
    """
    if len(poly) < 6:
        return None

    clipped = []
    for i in range(0, len(poly), 2):
        x = float(np.clip(poly[i],     xstart, xend))
        y = float(np.clip(poly[i + 1], ystart, yend))
        clipped.extend([x, y])

    return clipped if len(clipped) >= 6 else None
# =====================================================================
# Phase 2 — Category ID validation
# =====================================================================

def fix_category_id(cid, valid_ids, ann_id, stats, logger):
    """
    Validate or reject category_id.
    Returns:
        fixed ID or None: drop annotation.
    """
    if cid is None:
        stats["missing_category_id"] += 1
        logger(f"[SKIP] ann {ann_id}: missing category_id")
        return None

    if cid not in valid_ids:
        stats["invalid_category_ids"] += 1
        logger(f"[DROP] ann {ann_id}: invalid category_id={cid}")
        return None

    return cid


# =====================================================================
# Phase 3 — Deduplication (NMS-style)
# =====================================================================

def box_iou(b1, b2):
    """
    Compute IoU for COCO boxes [x, y, w, h].
    """
    x1, y1, w1, h1 = b1
    x2, y2 = x1 + w1, y1 + h1

    xx1, yy1, ww1, hh1 = b2
    xx2, yy2 = xx1 + ww1, yy1 + hh1

    inter_x1 = max(x1,  xx1)
    inter_y1 = max(y1,  yy1)
    inter_x2 = min(x2,  xx2)
    inter_y2 = min(y2,  yy2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = ww1 * hh1
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def deduplicate_annotations(anns, thresh=0.9):
    """
    Remove duplicated annotations with matching category_id and IoU > thresh.
    Deterministic order via sorting by annotation ID.
    """
    anns = sorted(anns, key=lambda a: a["id"])
    keep = []

    for ann in anns:
        is_dup = False
        for kept in keep:
            if ann["category_id"] == kept["category_id"]:
                if box_iou(ann["bbox"], kept["bbox"]) > thresh:
                    is_dup = True
                    break

        if not is_dup:
            keep.append(ann)

    return keep


# =====================================================================
# Phase 4 — Annotation consistency checks
# =====================================================================

def annotation_is_consistent(ann, image_dict, stats, logger):
    """
    Validate minimal required fields & references.
    """
    ann_id = ann.get("id")

    # Image ref required
    if "image_id" not in ann:
        stats["missing_image_id"] += 1
        logger(f"[SKIP] ann {ann_id}: missing image_id")
        return False

    if ann["image_id"] not in image_dict:
        stats["ann_ref_missing_image"] += 1
        logger(f"[SKIP] ann {ann_id}: references missing image_id={ann['image_id']}")
        return False

    # Bbox required
    if "bbox" not in ann:
        stats["missing_bbox"] += 1
        logger(f"[SKIP] ann {ann_id}: missing bbox")
        return False

    # Segmentation optional, but must be list format
    if "segmentation" in ann:
        if not isinstance(ann["segmentation"], list):
            stats["segmentation_invalid_format"] += 1
            logger(f"[SKIP] ann {ann_id}: invalid segmentation format")
            return False

    return True


# =====================================================================
# Phase 5 — Heuristic annotation fixes
# =====================================================================

def heuristic_fix_annotation(ann):
    """
    Add missing fields & normalize segmentation.
    """
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0

    if ann.get("segmentation") is None:
        ann["segmentation"] = []

    return ann


# =====================================================================
# Phase 7 — Main preprocessing pipeline
# =====================================================================

def preprocess_annotations(coco, valid_ids, logger=print):
    """
    Full COCO preprocessing pipeline:
      - consistency checks
      - category fix
      - bbox + polygon clipping
      - heuristic fixes
      - deduplication
    """

    stats = defaultdict(int)
    image_dict = {img["id"]: img for img in coco["images"]}
    cleaned = []

    # --------------------------------------------------------------
    # Main loop over annotations
    # --------------------------------------------------------------
    for ann in coco["annotations"]:
        ann_id = ann.get("id")

        # Phase 4: structural consistency
        if not annotation_is_consistent(ann, image_dict, stats, logger):
            continue

        img = image_dict[ann["image_id"]]
        W, H = img["width"], img["height"]

        # Phase 2: category validation
        cid = fix_category_id(ann.get("category_id"), valid_ids, ann_id, stats, logger)
        if cid is None:
            continue

        ann["category_id"] = cid

        # Phase 5: heuristic cleanup
        ann = heuristic_fix_annotation(ann)

        # Phase 1: clip bbox
        fixed_bbox = clip_box_to_image(ann["bbox"], W, H)
        if fixed_bbox is None:
            stats["invalid_bbox"] += 1
            logger(f"[SKIP] ann {ann_id}: invalid bbox after clipping")
            continue

        ann["bbox"] = fixed_bbox

        # Fix segmentation polygons
        new_segs = []
        for poly in ann.get("segmentation", []):
            clipped_poly = clip_polygon(poly, W - 1, H - 1)
            if clipped_poly:
                new_segs.append(clipped_poly)

        ann["segmentation"] = new_segs

        if len(new_segs) == 0:
            stats["segmentation_empty"] += 1
            logger(f"[SKIP] ann {ann_id}: segmentation empty after fixing")
            continue

        cleaned.append(ann)
        stats["kept"] += 1

    # --------------------------------------------------------------
    # Phase 3: Deduplicate final annotations
    # --------------------------------------------------------------
    logger("Running deduplication...")
    cleaned = deduplicate_annotations(cleaned, thresh=0.9)

    stats["final_count"] = len(cleaned)
    return cleaned, dict(stats)


# def postprocess_annotations():
#         # ======================================================
#         # annotation must be inside predicted ROI
#         # ======================================================
#         if ann_bbox is not None:
#             ax, ay, aw, ah = ann_bbox
#             ann_x1 = int(ax)
#             ann_y1 = int(ay)
#             ann_x2 = int(ax + aw)
#             ann_y2 = int(ay + ah)

#             clipped_ann_bbox = self.crop_ann_to_roi([x1, y1, x2, y2],
#                                  [ann_x1, ann_y1, ann_x2, ann_y2])
#             if self.is_zero_bbox(clipped_ann_bbox):
#                 # Annotation is way outside predicted ROI: reject
#                 self.rejected_out_of_boundaries_counter += 1
#                 self.rejected_out_of_boundaries_ids.append(img_id)
        
#                 return None if mode == 'training' else (
#                     [0,0,img_w,img_h], None, None, score
#                 )

def boxes_overlap(A, B, min_ratio=0.1):
    """
    Check if two XYXY-format boxes overlap with required overlap ratio.

    A: [Ax1, Ay1, Ax2, Ay2]    <-- bad box
    B: [Bx1, By1, Bx2, By2]    <-- ann box
    min_ratio: required overlap ratio relative to area(B)

    Returns True only if: intersection_area(A,B) > min_ratio * area(B)
    """

    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B

    # Check validity of boxes
    if Ax2 <= Ax1 or Ay2 <= Ay1:
        return False
    if Bx2 <= Bx1 or By2 <= By1:
        return False

    # Compute intersection coordinates
    ix1 = max(Ax1, Bx1)
    iy1 = max(Ay1, By1)
    ix2 = min(Ax2, Bx2)
    iy2 = min(Ay2, By2)

    # No intersection
    if ix2 <= ix1 or iy2 <= iy1:
        return False

    # Intersection area
    inter_area = (ix2 - ix1) * (iy2 - iy1)

    # Area of annotation box B
    B_area = (Bx2 - Bx1) * (By2 - By1)

    # Required: intersection > 10% of B
    return inter_area > (min_ratio * B_area)



def is_zero_bbox(bbox):
    x, y, w, h = bbox
    return (w == 0) or (h == 0)
    
def crop_mask(mask, roi):
    x1, y1, x2, y2 = map(int, roi)
    return mask[y1:y2, x1:x2]

def ann_to_mask(ann, height, width):
    seg = ann["segmentation"]
    if isinstance(seg, list):  # polygon
        rles = mask_utils.frPyObjects(seg, height, width)
        mask = mask_utils.decode(rles)
        return mask.max(axis=2)
    else:  # already RLE
        return mask_utils.decode(seg)
   
def mask_to_segmentation(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def crop_ann_bbox_to_roi(big_roi, small_box, f=0.65):
    """
    Crop annotation bbox so that it fits inside ROI.

    big_roi:  [x1, y1, w1, h1]
    small_box: [x1, y1, w2, h2]

    Rules:
    - Fully outside: return [0,0,0,0]
    - Partially inside: clipped to ROI
    - If clipped width < f * org width: [0,0,0,0]
    
    Returns a COCO-format bbox: [x, y, w, h]
    """
    bx1, by1, bw, bh = big_roi
    sx1, sy1, sw, sh = small_box
    
    org_sbox_area = (sw) * (sh)
    # Clip annotation into ROI
    cx1 = max(bx1, sx1)
    cy1 = max(by1, sy1)
    cx2 = min(bx1 + bw, sx1 + sw)
    cy2 = min(by1 + bh, sy1 + sh)
   
    # Completely outside
    if cx2 <= cx1 or cy2 <= cy1:
        return [0, 0, 0, 0]

    # Width / height
    cw = cx2 - cx1
    ch = cy2 - cy1
        

    # If too small: discard annotation
    if (cw * ch) < f * org_sbox_area:
        return [0, 0, 0, 0]

    # Valid clipped bbox in COCO format
    return [int(cx1), int(cy1), int(cw), int(ch)]

def compute_area(mask):
    return int(mask.sum())

def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None  # object removed
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

def update_annotation_after_roi_crop(ann, roi, orig_h, orig_w):
    """Modify ann in-place: update segmentation, bbox, area after cropping to ROI."""
    
    # Compute bbox
    bbox = crop_ann_bbox_to_roi(roi, ann["bbox"])
    if is_zero_bbox(bbox):
        return False
    
    # Fix segmentation mask
    new_segs = []
    if ann["segmentation"] is not None:
        mask = ann_to_mask(ann, orig_h, orig_w)
        
        cropped = crop_mask(mask, roi)
        
        # 3) If completely removed, return False so caller can delete annotation
        if cropped.sum() == 0:
            return False
        
        area = compute_area(cropped)
        
        # Update segmentation field
        x1, y1, w, h = map(int, roi)
        x2 = x1 + w
        y2 = x2 + h
        for poly in ann.get("segmentation", []):
            clipped_poly = clip_polygon(poly, x2, y2, x1, y1)
            if clipped_poly:
                new_segs.append(clipped_poly)
        
        if len(new_segs) == 0:
            return False    
    else:
        # Fall back to bbox area
        area = bbox[2] * bbox[3]
        
    # 4) Convert cropped mask to segmentation RLE
    ann["segmentation"] = new_segs

    # 5) Compute area
    ann["area"] = area


    ann["bbox"] = bbox  # [x,y,w,h] in NEW cropped image coords

    return True

# =====================================================================
# Utility: extract valid category IDs
# =====================================================================

def extract_valid_category_ids(coco):
    """
    Retrieve all category_id values from COCO JSON.
    """
    if "categories" not in coco:
        raise ValueError("COCO JSON missing 'categories' field.")

    return sorted(cat["id"] for cat in coco["categories"])


# =====================================================================
# Standalone CLI usage
# =====================================================================

if __name__ == "__main__":

    dataset = "val"
    data_dir = "../data/Dataset"
    ann_dir = os.path.join(data_dir, "annotations")
    ann_file = f"annotations_{dataset}"
    full_ann_path = os.path.join(ann_dir, f"{ann_file}.json")

    with open(full_ann_path, "r") as f:
        coco = json.load(f)

    valid_ids = extract_valid_category_ids(coco)
    logger = make_file_logger(f"../logs/preprocess_{ann_file}.log")

    cleaned_anns, stats = preprocess_annotations(coco, valid_ids, logger=logger)

    # Build new COCO file
    fixed_coco = {
        "licenses": coco["licenses"],
        "info": coco["info"],
        "images": coco["images"],
        "annotations": cleaned_anns,
        "categories": coco["categories"],
    }

    output_path = os.path.join(ann_dir, f"annotations_{dataset}_fixed.json")

    with open(output_path, "w") as f:
        json.dump(fixed_coco, f, indent=2)

    print(f"Saved updated COCO annotations to {output_path}")

    print("\nSUMMARY STATS:")
    for k, v in stats.items():
        print(f"{k:30s}: {v}")
