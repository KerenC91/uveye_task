import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class ROICropper:
    # Vehicle classes in COCO
    VEHICLE_CLASSES = {2, 7, 5, 3}   # car, truck, bus, motorcycle

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

    def extract_roi(self, 
                    bboxes, scores, labels, 
                    img_w, img_h, 
                    th=0.5, 
                    mode='training',
                    min_area_ratio=0.05,
                    max_area_ratio=1.0):
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
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, 0.0)
    
        # ======================================================
        # 2. Filter only vehicle classes
        # ======================================================
        valid_indices = [i for i, lbl in enumerate(labels) 
                         if lbl in ROICropper.VEHICLE_CLASSES]
        if len(valid_indices) == 0:
            self.full_area_counter += 1
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, 0.0)
    
        # ======================================================
        # 3. Pick highest-confidence bounding box
        # ======================================================
        valid_bboxes = bboxes[valid_indices]
        valid_scores = scores[valid_indices]
        valid_labels = labels[valid_indices]
    
        idx = np.argmax(valid_scores)
        x1, y1, x2, y2 = valid_bboxes[idx]
        cls = int(valid_labels[idx])
        score = float(valid_scores[idx])
        class_name = self.class_names[cls] if self.class_names else None
    
        # ======================================================
        # 4. Score threshold check
        # ======================================================
        if score < th:
            self.rejected_low_score += 1
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, score)
    
        # ======================================================
        # 5. Fix bounding box numeric issues (clip + enforce ordering)
        # ======================================================
        # Clip to image boundaries
        x1 = max(0, min(int(x1), img_w - 1))
        x2 = max(0, min(int(x2), img_w))
        y1 = max(0, min(int(y1), img_h - 1))
        y2 = max(0, min(int(y2), img_h))
    
        # Enforce x1 < x2 and y1 < y2
        if x2 <= x1 or y2 <= y1:
            self.rejected_zero_size += 1
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, score)
    
        # ======================================================
        # 6. Validate Area Ratio
        # ======================================================
        area = (x2 - x1) * (y2 - y1)
        area_ratio = area / (img_w * img_h)
    
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            self.rejected_area_ratio += 1
            return None if mode == 'training' else ([0, 0, img_w, img_h], None, None, score)
    
        # ======================================================
        # 7. ROI is VALID
        # ======================================================
        roi = [x1, y1, x2, y2]
    
        self.class_counter[class_name] += 1
        self.class_scores_sum[class_name] += score
        self.global_scores.append(score)
        self.accepted_counter += 1
    
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
            "rejected_zero_size": self.rejected_zero_size
        }
    
    def show_hist(self, filename):
        plt.hist(self.global_scores, bins=20)
        plt.title("Score distribution")
        plt.savefig(filename)
        plt.show()
        
    
