_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

root = '/scratch/home/kerencohen2/uveye_task/car_defects_detection/mmdetection'
data_root = root + '/data/Dataset'

train_batch_size_per_gpu = 2
train_num_workers = 0#2

max_epochs = 20#20
stage2_num_epochs = 1
base_lr = 0.00008#0.02?
dataset_type = 'CocoDatasetWithROI'
n_classes = 6
backend_args = None
load_from = root + '/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

metainfo = {
'classes': ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat', ),
'palette': [
        (255, 105, 180),   # pink       
        (135, 206, 250),   # light blue 
        (0, 255, 0),       # green       
        (160, 32, 240),    # purple      
        (255, 255, 0),     # yellow      
        (255, 0, 0),       # red       
    ]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='CropToCarROI'),
#    dict(type = 'RandomCrop', crop_size=(640, 640)),
    dict(
        type='Resize',
        scale=(1333, 800), #how to choose that?
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='Resize',
        scale=(1333, 800), #how to choose that?
        keep_ratio=True),
    dict(type='PackDetInputs',
	    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
			   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers = False,
    sampler=dict(type='DefaultSampler',shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
    	type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train2017/'),
        ann_file=data_root + '/annotations/annotations_train_postprocessroi.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
        
val_dataloader = dict(
	batch_size=1,
	num_workers=train_num_workers,
	persistent_workers = False,
	sampler=dict(type='DefaultSampler',shuffle=False),
	dataset=dict(
		type=dataset_type,
		data_root=data_root,
		metainfo=metainfo,
		data_prefix=dict(img='val2017/'),
		test_mode=True,
		ann_file=data_root + '/annotations/annotations_val_postprocessroi.json',
		pipeline=test_pipeline,
		backend_args=backend_args))

test_dataloader = dict(
	batch_size=1,
	num_workers=train_num_workers,
	persistent_workers = False,
	sampler=dict(type='DefaultSampler',shuffle=False),
	dataset=dict(
	type=dataset_type,
		data_root=data_root,
		metainfo=metainfo,
		data_prefix=dict(img='test2017/'),
		test_mode=True,
		ann_file=data_root + '/annotations/annotations_test_postprocessroi.json',
		pipeline=test_pipeline,
		backend_args=backend_args))
        
val_evaluator = dict(ann_file=data_root + '/annotations/annotations_val_postprocessroi.json', metric=['bbox', 'segm'], format_only=False, backend_args=backend_args)

test_evaluator = dict(ann_file=data_root + '/annotations/annotations_test_postprocessroi.json', metric=['bbox', 'segm'], format_only=True, 
		outfile_prefix='./work_dirs/car_defects_detection/test', backend_args=backend_args)
#####


model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=n_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=n_classes,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))


# optimizer
optim_wrapper = dict(
	_delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05))

amp = dict(type='AMPOptimizerHook')

param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=500),  
    # The main LRScheduler
    dict(
        type='MultiStepLR',  
        by_epoch=True,  
        begin=0,   
        end=20,  
        milestones=[14, 18],  
        gamma=0.1) 
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'), 
    checkpoint=dict(
        interval=50,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    logger=dict(type='LoggerHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
    )
#resume = True
custom_hooks = []

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
