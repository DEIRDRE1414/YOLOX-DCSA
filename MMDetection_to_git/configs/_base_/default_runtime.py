default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,                                           # 是否启用 cudnn benchmark
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),       # 使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能存在隐患。请参考 https://github.com/pytorch/pytorch/issues/1355 。关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend='nccl'),                                   ##分布式相关设置
)
# 可视化检测结果，在拼接图像中，其中左侧图像是ground truth，右侧图像是预测。
# 定义在MMDET.VISUALIZATION.LOCAL_VISUALIZER
#- 如果 show 为 True，则忽略所有存储后端，图像将显示在本地窗口中。
#- 如果指定out_file，绘制的图像将保存到out_file,通常在显示器不可用时使用。


#vis_backends = [dict(type='LocalVisBackend')]
vis_backends = [dict(type='TensorboardVisBackend')]                  # 可视化后端

#vis_backends = [dict(type='WandbVisBackend')]

#visualizer = dict(
    #type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)    # 日志处理器用于处理运行时日志# 日志数值的平滑窗口# 是否使用 epoch 格式的日志。需要与训练循环的类型保存一致。

log_level = 'INFO'                       # 日志等级
load_from = None                         # 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False                           # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。
