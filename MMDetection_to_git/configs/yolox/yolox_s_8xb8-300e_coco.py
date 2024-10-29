_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './yolox_tta.py'
]
img_scale = (640, 640)  # width, height

# model settings
model = dict(
    type='YOLOX',                                # 检测器名
    data_preprocessor=dict(                      # 数据预处理器的配置，通常包括图像归一化和 padding
        type='DetDataPreprocessor',              # 数据预处理器的类型，参考
        pad_size_divisor=32,                     # padding 后的图像的大小应该可以被 ``pad_size_divisor`` 整除,图像 padding 到 32 的倍数
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',    # 批量同步随机调整大小，用于将批量中的图像按照相同的比例进行随机调整大小
                random_size_range=(480, 800),    # 多尺度范围是 480~800
                size_divisor=32,                  # 输出尺度需要被 32 整除
                interval=10)                      # 每隔 10 个迭代改变一次输出输出
        ]),
    backbone=dict(
        type='CSPDarknet',                       # 主干网络的类别，可用选项请参考
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),                   # 每个状态产生的特征图输出的索引
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),     # 归一化层(norm layer)的配置项，归一化层的类别，通常是 BN 或 GN
        act_cfg=dict(type='Swish'),                             # 激活函数(activation function)的配置项
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],             # 骨架多尺度特征图输出通道，输入通道数，这与主干网络的输出通道一致
        out_channels=128,                        # 金字塔特征图每一层的输出通道
        num_csp_blocks=1,                        # CSPLayer 中 bottlenecks 的数量
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        #conv_cfg=dict(type='SahiConv2d'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=11,
        in_channels=128,                       # 每个输入特征图的输入通道，这与 neck 的输出通道一致
        feat_channels=128,                     # head 卷积层的特征通道
        stacked_convs=2,
        strides=(8, 16, 32),                   # 锚生成器的步幅。这与 FPN 特征步幅一致。 如果未设置 base_sizes，则当前步幅值将被视为 base_sizes
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(                         # 分类分支的损失函数配置
            type='FocalLoss',
            #type='CrossEntropyLoss',           # 分类分支的损失类型，我们也支持 FocalLoss 等
            use_sigmoid=True,                  # RPN 通常进行二分类，所以通常使用 sigmoid 函数
            #alpha=0.10,
            reduction='sum',
            loss_weight=1.0),                  # 分类分支的损失权重
        loss_bbox=dict(                        # 回归分支的损失函数配置
            type='CIoULoss',
            #type='IoULoss',
            #mode='square',
            #eps=1e-16,
            eps=1e-7,
            reduction='sum',
            loss_weight=5.0),

        loss_obj=dict(
            type='FocalLoss',
            #type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=5.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
    #  0.01过滤类别的分值，低于 score_thr 的检测框当做背景处理
    # NMS 的类型和阈值
# dataset settings
data_root = 'data1/coco/'
dataset_type = 'CocoDataset'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [

    #dict(type='ClassAwareSampler', img_scale=img_scale, num_sample_class=1),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),   # Mosaic 数据增强方法，# Mosaic 数据增强后的图像尺寸，img_scale 顺序应该是 (width, height)
    dict(
        type='RandomAffine',                                   # YOLOv5 的随机仿射变换
        scaling_ratio_range=(0.8, 1.3),                          # 图像缩放系数的范围
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),      # 从输入图像的高度和宽度两侧调整输出形状的距离,图像经过马赛克处理后会放大4倍，所以我们使用仿射变换来恢复图像的大小
    dict(type='Rotate', prob=0.5),
    #dict(type='Sharpness', prob=0.5),
    dict(
        type='MixUp',
        #img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),                            # HSV通道随机增强
    #dict(type='CutOut', n_holes=(1, 3), cutout_shape=None, cutout_ratio=(0.2, 0.4), fill_in=(0, 0, 0)),
    dict(type='RandomFlip', prob=0.5),                         # 随机翻转，翻转概率 0.5
    #dict(type='Sharpness', prob=0.5),
    #dict(type='Rotate', prob=0.5),

    #dict(type='MinIoURandomCrop', min_ious=(0.3, ), min_crop_size=0.1),
    #dict(
        #type='RandomOrder',
        #transforms=[
            #dict(type='YOLOXHSVRandomAug'),
            #dict(type='RandomFlip', prob=0.5),
            #dict(type='Sharpness', prob=0.1,max_mag=0.5),
            #dict(type='MinIoURandomCrop', min_ious=(0.9, ), min_crop_size=0.1),

        #],
   # ),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')                                # 将数据转换为检测器输入格式的流程
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    #_delete_=True,
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',                 # 标注文件路径
        data_prefix=dict(img='train2017/'),                              # 图像路径前缀
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
            #dict(type='ClassAwareSampler',  num_sample_class=1)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),             # 图像和标注的过滤配置
        backend_args=backend_args),
    pipeline=train_pipeline)                                             # 这是由之前创建的 train_pipeline 定义的数据处理流程

test_pipeline = [                                                        # 测试数据处理流程
    dict(type='LoadImageFromFile', backend_args=backend_args),           # 第 1 个流程，从文件路径里加载图像
    #img_scale = (1024, 1024)
    dict(type='Resize', scale=img_scale, keep_ratio=True),               # 第 2 个流程，保持长宽比的图像大小缩放，# 图像缩放的目标尺寸
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),                        # 第 4 个流程，对于当前图像，加载它的注释信息
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',       # 将数据转换为检测器输入格式的流程
                   'scale_factor'))
]

train_dataloader = dict(                                                 # 训练 dataloader 配置
    #_delete_=True,
    #type=dataset_type,
    batch_size=16,
    num_workers=6,
    persistent_workers=True,                    # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    #sampler=dict(type='DefaultSampler', shuffle=True),                   # 默认的采样器，同时支持分布式和非分布式训练，shuffle随机打乱每个轮次训练数据的顺序
    sampler=dict(type='ClassAwareSampler',num_sample_class=11),
    #batch_sampler=dict(type='AspectRatioBatchSampler'),
    #ClassAwareSampler=dict(type='ClassAwareSampler'),
    #class_aware_sampler=dict(num_sample_class=1),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,                                                    # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(type='DefaultSampler', shuffle=False),                 # 默认的采样器，同时支持分布式和非分布式训练
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',                  # 标注文件路径
        data_prefix=dict(img='val2017/'),                               # 图像路径前缀
        #ann_file='annotations/instances_test2017.json',
        #data_prefix=dict(img='test2017/'),

        test_mode=True,                                                 # 开启测试模式，避免数据集过滤图像和标注
        pipeline=test_pipeline,                                         # 这是由之前创建的 test_pipeline 定义的数据处理流程
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(                                                   # 验证过程使用的评测器
    type='CocoMetric',                                                  # 用于评估检测的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=data_root + 'annotations/instances_val2017.json',
    #ann_file=data_root + 'annotations/instances_test2017.json',
    metric='bbox',                                                      # 需要计算的评价指标，`bbox` 用于检测
    backend_args=backend_args)
test_evaluator = val_evaluator                                          # 测试过程使用的评测器

# training settings
max_epochs = 70
num_last_epochs = 15
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.01                                                        # 基础学习率
optim_wrapper = dict(                                                 # 优化器封装的配置
    #type='OptimWrapper',                                              # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,      # 随机梯度下降优化器，momentum 带动量的随机梯度下降，权重衰减
        nesterov=True),                                               # 开启Nesterov momentum，公式详见 http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        #interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',                                         # 实现权重 EMA(指数移动平均) 更新的 Hook
        ema_type='ExpMomentumEMA',                              # YOLO 中使用的带动量 EMA
        momentum=0.0001,                                        # EMA 的动量参数
        update_buffers=True,                                    # 是否计算模型的参数和缓冲的 running averages
        priority=49)                                            # 优先级略高于 NORMAL(50)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
