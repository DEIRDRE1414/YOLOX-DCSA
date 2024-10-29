_base_ = '../yolox/yolox_s_8xb8-300e_coco.py'

img_scale = (640, 640)

data_root = 'data1/coco/'
dataset_type = 'CocoDataset'
#model = dict(mask_head=dict(num_classes=3))
model = dict(bbox_head=dict(num_classes=11))

#bbox_head=dict(num_classes=11)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    #logger=dict(type='LoggerHook', interval=5),
    logger=dict(type='LoggerHook', interval=50),                 # 日志打印
    param_scheduler=dict(type='ParamSchedulerHook'),            # 参数（学习率等）调度
    checkpoint=dict(type='CheckpointHook', interval=10),         # 保存checkpoint权重
    sampler_seed=dict(type='DistSamplerSeedHook'),              # 设置数据采样的随机种子，确保shuffle生效
    visualization=dict(type='DetVisualizationHook'))             # 用于可视化验证或测试过程的预测结果，定义在mmdet中

#data = dict(
#train_dataloader=dict(class_aware_sampler=dict(num_sample_class=1))
#train_pipeline = [dict(type='ClassAwareSampler', img_scale=img_scale, num_sample_class=1),]
max_epochs = 70


#visualization = _base_.default_hooks.visualization
#visualization.update(dict(draw=True, show=True))

load_from = 'D:/doctor/MMDetection/configs/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
