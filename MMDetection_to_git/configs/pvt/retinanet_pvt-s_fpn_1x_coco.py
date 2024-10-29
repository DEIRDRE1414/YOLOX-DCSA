_base_ = 'retinanet_pvt-t_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='configs/pvt/retinanet_pvt-s_fpn_1x_coco_20210906_142921-b6c94a5b.pth')))
