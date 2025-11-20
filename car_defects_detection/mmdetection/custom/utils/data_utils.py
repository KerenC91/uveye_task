import random
import cv2
from mmdet.apis import DetInferencer


def get_model_config(model_key: str, model_zoo):
    """
    Extract model configuration given a model name.
    """
    if model_key not in model_key:
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {list(model_zoo.keys())}"
        )
    cfg = model_zoo[model_key]
    return cfg["model_name"], cfg["checkpoint"]

# --------------------------------------------------------------------
# Image Subset Handling
# --------------------------------------------------------------------
def select_image_ids(images, num):
    """
    Select full or random subset of image IDs.
    """
    all_ids = [img["id"] for img in images]
    return random.sample(all_ids, num) if num > 0 else all_ids


# --------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------
def load_model(model_key, model_zoo):
    """
    Load detector model and class names.
    """
    model_name, checkpoint = get_model_config(model_key, model_zoo)
    inferencer = DetInferencer(model_name, checkpoint)
    class_names = inferencer.model.dataset_meta["classes"]
    return inferencer, class_names

def load_image_rgb(path):
    """Load image safely as RGB."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_valid_category_ids(coco):
    """
    Retrieve all category_id values from COCO JSON.
    """
    if "categories" not in coco:
        raise ValueError("COCO JSON missing 'categories' field.")

    return sorted(cat["id"] for cat in coco["categories"])