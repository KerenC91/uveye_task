import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class ROICropper:
    # Vehicle classes in COCO
    VEHICLE_CLASSES = {2, 7, 5, 3}#, 8, 4, 6}   # car, truck, bus, motorcycle, boat, airplane, train

    def __init__(self, class_names=None):
        self.class_counter = defaultdict(int)
        self.full_area_counter = 0
        self.rejected_low_score = 0
        self.accepted_counter = 0
        self.rejected_area_ratio = 0
        self.rejected_zero_size = 0
        self.class_scores_sum = defaultdict(float)
        self.global_scores = []
        self.class_names = class_names
        self.full_area_ids = []
        self.rejected_low_score_ids = []
        self.accepted_ids = []
        self.rejected_area_ratio_ids = []
        self.rejected_zero_size_ids = []


    def extract_roi(self, 
                    bboxes, scores, labels, 
                    img_w, img_h, img_id,
                    th=0.5, 
                    mode='training',
                    min_area_ratio=0.05,
                    max_area_ratio=1.0,
                    dominance_ratio=1.8):
        """
        bboxes: Nx4 array [x1, y1, x2, y2]
        scores: Nx array
        labels: Nx array
        mode: 
            'training'  -> discard invalid ROIs
            'inference' -> fallback to full image
        """
    
        # ======================================================
        # 1. Handle case: NO predictions at all
        # ======================================================
        if bboxes is None or len(bboxes) == 0:
            self.full_area_counter += 1
            self.full_area_ids.append(img_id)
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, 0.0)
    
        # ======================================================
        # 2. Filter only vehicle classes
        # ======================================================
        valid_indices = [i for i, lbl in enumerate(labels) if lbl in ROICropper.VEHICLE_CLASSES]
    
        if len(valid_indices) == 0:
            self.full_area_counter += 1
            self.full_area_ids.append(img_id)
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, 0.0)
    
        # ======================================================
        # 3. Filter by score first
        # ======================================================
        valid_bboxes = bboxes[valid_indices]
        valid_scores = scores[valid_indices]
        valid_labels = labels[valid_indices]
    
        score_mask = valid_scores >= th
        if not np.any(score_mask):
            self.rejected_low_score += 1
            self.rejected_low_score_ids.append(img_id)
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, 0.0)
    
        bboxes_thr = valid_bboxes[score_mask]
        scores_thr = valid_scores[score_mask]
        labels_thr = valid_labels[score_mask]
    
        # ======================================================
        # 4. Compute areas of *all* thresholded detections
        # ======================================================
        b = bboxes_thr.copy()
        b[:, 0] = np.clip(b[:, 0], 0, img_w - 1)   # x1
        b[:, 2] = np.clip(b[:, 2], 0, img_w - 1)   # x2
        b[:, 1] = np.clip(b[:, 1], 0, img_h - 1)   # y1
        b[:, 3] = np.clip(b[:, 3], 0, img_h - 1)   # y2
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        sorted_indices = np.argsort(-areas)  # descending
    
        largest_idx = sorted_indices[0]
        largest_area = areas[largest_idx]
    
        # # Determine second largest area
        # if len(sorted_indices) > 1:
        #     second_area = areas[sorted_indices[1]]
        # else:
        #     second_area = 0
    
        # # Check dominance constraint
        # if second_area > 0 and largest_area < dominance_ratio * second_area:
        #     # Foreground not dominant enough → danger of picking wrong car
        #     self.rejected_area_ratio += 1
        #     self.rejected_area_ratio_ids.append(img_id)
        #     return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, 0.0)
    
        # ======================================================
        # 5. Get ROI from largest-area box
        # ======================================================
        x1, y1, x2, y2 = b[largest_idx]
        cls = int(labels_thr[largest_idx])
        score = float(scores_thr[largest_idx])
        class_name = self.class_names[cls] if self.class_names else None
    
        # ------------------------------------------------------
        # Clip bounding box & ensure valid size
        # ------------------------------------------------------
        x1 = max(0, min(int(x1), img_w - 1))
        x2 = max(0, min(int(x2), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        y2 = max(0, min(int(y2), img_h - 1))
    
        if x2 <= x1 or y2 <= y1:
            self.rejected_zero_size += 1
            self.rejected_zero_size_ids.append(img_id)
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, score)
    
        # ======================================================
        # 6. Validate Area Ratio relative to image
        # ======================================================
        area_ratio = largest_area / (img_w * img_h)
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            self.rejected_area_ratio += 1
            self.rejected_area_ratio_ids.append(img_id)
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, score)
    
        # ======================================================
        # 7. ROI is VALID
        # ======================================================
        roi = [x1, y1, x2, y2]
    
        self.class_counter[class_name] += 1
        self.class_scores_sum[class_name] += score
        self.global_scores.append(score)
        self.accepted_counter += 1
        self.accepted_ids.append(img_id)
    
        return roi, cls, class_name, score




    def get_statistics(self):
        per_class_stats = {}

        for class_name in self.class_counter:
            count = self.class_counter[class_name]
            avg_score = self.class_scores_sum[class_name] / count
            per_class_stats[class_name] = {
                "count": count,
                "avg_accuracy": avg_score
            }

        global_avg = float(np.mean(self.global_scores)) if self.global_scores else 0.0
        global_min = float(np.min(self.global_scores)) if self.global_scores else 0.0

        return {
            "per_class": per_class_stats,
            "global_avg_accuracy": global_avg,
            "global_min_accuracy": global_min,
            "full_area_counter": self.full_area_counter,
            "rejected_low_score": self.rejected_low_score,
            "accepted_counter": self.accepted_counter, 
            "rejected_area_ratio": self.rejected_area_ratio,
            "rejected_zero_size": self.rejected_zero_size,
            "full_area_ids": self.full_area_ids,
            "rejected_low_score_ids": self.rejected_low_score_ids,
            "accepted_ids": self.accepted_ids,
            "rejected_area_ratio_ids": self.rejected_area_ratio_ids,
            "rejected_zero_size_ids": self.rejected_zero_size_ids
        }
    
    def show_hist(self, filename):
        plt.hist(self.global_scores, bins=20)
        plt.title("Score distribution")
        plt.savefig(filename)
        plt.show()
        
    
