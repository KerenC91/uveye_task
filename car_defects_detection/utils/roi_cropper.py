import numpy as np
from collections import defaultdict

import numpy as np
from collections import defaultdict

class ROICropper:
    # Vehicle classes in COCO
    VEHICLE_CLASSES = {2, 7, 5, 3}   # car, truck, bus, motorcycle

    def __init__(self, class_names=None):
        self.class_counter = defaultdict(int)
        self.class_scores_sum = defaultdict(float)
        self.global_scores = []
        self.class_names = class_names

    def extract_roi(self, bboxes, scores, labels, img_w, img_h, th=0.5):
        """
        bboxes: Nx4 array [x1, y1, x2, y2]
        scores: Nx array
        labels: Nx array
        """
    
        # No predictions at all → full image
        if bboxes is None or len(bboxes) == 0:
            return [0, 0, img_w, img_h], None, None, 0.0
    
        # Keep only vehicle classes
        valid_indices = [i for i, lbl in enumerate(labels) if lbl in ROICropper.VEHICLE_CLASSES]
    
        # If nothing relevant → full image
        if len(valid_indices) == 0:
            return [0, 0, img_w, img_h], None, None, 0.0
    
        # Extract only valid predictions
        valid_bboxes = bboxes[valid_indices]
        valid_scores = scores[valid_indices]
        valid_labels = labels[valid_indices]
    
        # Choose highest-confidence box
        idx = np.argmax(valid_scores)
    
        best_bbox = valid_bboxes[idx].tolist()
        cls = int(valid_labels[idx])
        score = float(valid_scores[idx])
        class_name = self.class_names[cls] if self.class_names else None
    
        # If confidence is too low → fallback to full image
        if score < th:
            return [0, 0, img_w, img_h], None, None, score
    
        # Otherwise, accept ROI and update statistics
        self.class_counter[class_name] += 1
        self.class_scores_sum[class_name] += score
        self.global_scores.append(score)
    
        return best_bbox, cls, class_name, score



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
            "global_min_accuracy": global_min
        }

