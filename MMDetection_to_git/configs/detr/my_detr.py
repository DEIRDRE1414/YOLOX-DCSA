_base_ = '../detr/detr_r50_8xb2-150e_coco.py'

data_root = 'data/coco/'
dataset_type = 'CocoDataset'

#bbox_head=dict(num_classes=11)
model = dict(bbox_head=dict(num_classes=1))

train_dataloader = dict(batch_size=2)

load_from = 'C:/Users/关宁/MMDetection/configs/detr/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'