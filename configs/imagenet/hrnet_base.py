# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='HighResolutionNetOri',
        cfg={'STAGE1': {'NUM_MODULES': 1,
                        'NUM_RANCHES': 1,
                        'BLOCK': 'BOTTLENECK',
                        'NUM_BLOCKS': [2],
                        'NUM_CHANNELS': [64],
                        'FUSE_METHOD': 'SUM'},
             'STAGE2': {'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2],
                        'NUM_CHANNELS': [18, 36],
                        'FUSE_METHOD': 'SUM'},
             'STAGE3': {'NUM_MODULES': 3,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2],
                        'NUM_CHANNELS': [18, 36, 72],
                        'FUSE_METHOD': 'SUM'},
             'STAGE4': {'NUM_MODULES': 2,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2, 2],
                        'NUM_CHANNELS': [18, 36, 72, 144],
                        'FUSE_METHOD': 'SUM'},
             }
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

# optimizer 32 GPUs, 32 * 64
optimizer = dict(
    type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25,
    step=[30, 60, 90])
total_epochs = 100

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]