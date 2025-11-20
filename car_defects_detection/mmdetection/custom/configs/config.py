class Params:
    """Simple attribute-style configuration container."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


params = Params(
    # ------------------------------------------------------------
    # Dataset settings
    # ------------------------------------------------------------
    num_images=0,                  # 0: process all images

    # ------------------------------------------------------------
    # Model + inference settings
    # ------------------------------------------------------------
    mode='discard',               # fordiscard - discard rejects, for keep - keep with original roi 
    score_threshold=0.2,           # detection confidence threshold
    bad_threshold=0.5,             # for person class

    # ------------------------------------------------------------
    # ROI validation settings
    # ------------------------------------------------------------
    min_area_ratio=0.05,           # ROI must cover ≥ 5% of image
    max_area_ratio=1.0,            # ROI must not exceed full image

    # ------------------------------------------------------------
    # COCO class IDs considered "vehicles"
    # (car=2, motorcycle=3, bus=5, truck=7, boat=8, airplane=4)
    # ------------------------------------------------------------
    vehicle_classes={2, 3, 5, 7, 8, 4},
    invalid_classes={0}, # person
    # ------------------------------------------------------------
    # Model Zoo
    # ------------------------------------------------------------
    model_zoo={
        "yolo_s": {
            "model_name": "yolox_s_8x8_300e_coco",
            "checkpoint": (
                "../checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
            ),
        },
        "yolo_x": {
            "model_name": "yolox_x_8x8_300e_coco",
            "checkpoint": (
                "../checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
            ),
        },
        "rtmdet_tiny": {
            "model_name": "rtmdet_tiny_8xb32-300e_coco",
            "checkpoint": (
                "../checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
            ),
        },
        "rtmdet_x": {
            "model_name": "rtmdet_x_8xb32-300e_coco",
            "checkpoint": (
                "../checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
            ),
        },
    },
)

test_params = Params(
    # ------------------------------------------------------------
    # Dataset settings
    # ------------------------------------------------------------
    num_images=0,                  # 0: process all images

    # ------------------------------------------------------------
    # Model Zoo
    # ------------------------------------------------------------
    model_zoo={
        "rtmdet_tiny": {
            "model_name": "../configs/rtmdet/rtmdet_tiny_8xb32-300e_car_defects.py",
            "checkpoint": (
                "../work_dirs/rtmdet_tiny_8xb32-300e_car_defects/best_coco_bbox_mAP_epoch_17.pth"
            ),
        },
        "mask_rcnn_r50": {
            "model_name": "../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_car_defects.py",
            "checkpoint": (
                "../work_dirs/mask-rcnn_r50_fpn_1x_car_defects/best_coco_bbox_mAP_epoch_16.pth"
            ),
        },
    },
)
