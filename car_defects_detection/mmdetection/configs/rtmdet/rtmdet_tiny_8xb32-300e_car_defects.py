
root = '/scratch/home/kerencohen2/uveye_task/car_defects_detection/mmdetection'
_base_ = './rtmdet_s_8xb32-300e_coco.py'
data_root = root + '/data/Dataset'

train_batch_size_per_gpu = 4
train_num_workers = 0#2

max_epochs = 1#20
stage2_num_epochs = 1
base_lr = 0.00008
dataset_type = 'CocoDatasetWithROI'
n_classes = 6

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
    
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers = False,
    dataset=dict(
    	type='CocoDatasetWithROI',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train2017/'),
        ann_file=data_root + '/annotations/annotations_train_postprocess.json',
        filter_cfg=dict(filter_empty_gt=True),))
        
val_dataloader = dict(
    num_workers=train_num_workers,
    persistent_workers = False,
    dataset=dict(
    	type='CocoDatasetWithROI',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val2017/'),
        ann_file=data_root + '/annotations/annotations_val_postprocess.json'))

test_dataloader = dict(
    num_workers=train_num_workers,
    persistent_workers = False,
    dataset=dict(
    	type='CocoDatasetWithROI',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='test2017/'),
        ann_file=data_root + '/annotations/annotations_test_postprocess.json'))

val_evaluator = dict(ann_file=data_root + '/annotations/annotations_val_postprocess.json', metric=['bbox'],)#, 'segm'

test_evaluator = dict(ann_file=data_root + '/annotations/annotations_test_postprocess.json', metric=['bbox'],)

#checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

# load COCO pre-trained weight
load_from = root + '/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

model = dict(
    backbone=dict(
	frozen_stages=0,
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=load_from)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False, num_classes=n_classes))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
    dict(type='CropToCarROI'),

    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='PackDetInputs')
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
