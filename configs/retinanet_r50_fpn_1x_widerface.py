_base_ = [
    '../third_party/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '../third_party/mmdetection/configs/_base_/schedules/schedule_1x.py', 
    '../third_party/mmdetection/configs/_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)

classes = ('face',)
model = dict(bbox_head=dict(num_classes=1))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/widerface/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800,400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800,400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/train_coco.json',
        img_prefix=data_root + 'WIDER_train/images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val_coco.json',
        img_prefix=data_root + 'WIDER_val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/val_coco.json',
        img_prefix=data_root + 'WIDER_val/images',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
