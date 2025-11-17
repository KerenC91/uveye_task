class Params:
    """Simple attribute-style configuration container."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


params = Params(

    # ------------------------------------------------------------
    # Dataset settings
    # ------------------------------------------------------------
    data_dir="data/Dataset",
    dataset="val",                 # options: "train", "val", "test"
    use_fixed_ann=True,            # load annotations_*_fixed.json if True
    num_images=0,                  # 0: process all images

    # ------------------------------------------------------------
    # Model + inference settings
    # ------------------------------------------------------------
    model_key="rtmdet_x",            # choose from model_zoo keys
    score_threshold=0.0,           # detection confidence threshold
    save_debug_vis=True,           # save visualization and ROI jpeg

    # ------------------------------------------------------------
    # Output settings
    # ------------------------------------------------------------
    output_root="output_rtmdet_x_th_0.0",

    # ------------------------------------------------------------
    # ROI validation settings
    # ------------------------------------------------------------
    min_area_ratio=0.05,           # ROI must cover ≥ 5% of image
    max_area_ratio=1.0,            # ROI must not exceed full image

    # ------------------------------------------------------------
    # COCO class IDs considered "vehicles"
    # (car=2, motorcycle=3, bus=5, truck=7)
    # ------------------------------------------------------------
    vehicle_classes={2, 3, 5, 7},

    # ------------------------------------------------------------
    # Model Zoo
    # ------------------------------------------------------------
    model_zoo={
        "yolo_s": {
            "model_name": "yolox_s_8x8_300e_coco",
            "checkpoint": (
                "checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
            ),
        },
        "yolo_x": {
            "model_name": "yolox_x_8x8_300e_coco",
            "checkpoint": (
                "checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
            ),
        },
        "rtmdet_tiny": {
            "model_name": "rtmdet_tiny_8xb32-300e_coco",
            "checkpoint": (
                "checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
            ),
        },
        "rtmdet_x": {
            "model_name": "rtmdet_x_8xb32-300e_coco",
            "checkpoint": (
                "checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
            ),
        },
        # Add more models here if needed...
    },
)
