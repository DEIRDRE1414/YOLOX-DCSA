# 数据集准备
```
训练和测试的数据集文件夹结构
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_test2017.json
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

```
COCO标注格式
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
```

``` 数据集转换COCO格式
import os.path as osp

import mmcv

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress


def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'balloon'
        }])
    dump(coco_format_json, out_file)


if __name__ == '__main__':
    convert_balloon_to_coco(ann_file='data/balloon/train/via_region_data.json',
                            out_file='data/balloon/train/annotation_coco.json',
                            image_prefix='data/balloon/train')
    convert_balloon_to_coco(ann_file='data/balloon/val/via_region_data.json',
                            out_file='data/balloon/val/annotation_coco.json',
                            image_prefix='data/balloon/val')
```
本实验数据集收集自Google Earth和一些开源数据集，获取包含飞机类别的遥感图像，构建以飞机为目标的遥感数据集。使用图像标注工具对图像中的飞机目标进行标注，为每个目标绘制边界框，并记录其类别和位置信息。构建的数据集总共包含5 174张训练集，2 301张验证集。将数据集按 COCO 格式整理。由于数据集较大，github上未上传，如有需求请邮件联系作者。
# 准备配置文件

```
## 从 `_base_ = '../yolox/yolox_s_8xb8-300e_coco.py' ` 中继承的配置信息来构建模型的基本结构。
_base_ = '../yolox/yolox_s_8xb8-300e_coco.py'    
img_scale = (640, 640)    
data_root = 'data1/coco/'  
dataset_type = 'CocoDataset'
model = dict(bbox_head=dict(num_classes=11))
default_hooks = dict(  
    timer=dict(type='IterTimerHook'),   
    logger=dict(type='LoggerHook', interval=50),       
    param_scheduler=dict(type='ParamSchedulerHook'),   
    checkpoint=dict(type='CheckpointHook', interval=10), 
    sampler_seed=dict(type='DistSamplerSeedHook'),     
    visualization=dict(type='DetVisualizationHook'))   
max_epochs = 70
load_from = 'D:/doctor/MMDetection/configs/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
```
# 模型训练
##  单 GPU 训练
```
python tools/train.py /configs/yolox/my_yolox.py
```

## CPU 训练
```
 export CUDA_VISIBLE_DEVICES=-1
 python tools/train.py /configs/yolox/my_yolox.py
```

# 模型验证
## 单 GPU 验证
```
python tools/test.py configs/yolox/my_yolox.py work_dirs/my_yolox/epoch_12.pth
```
## CPU 验证
```
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py configs/yolox/my_yolox.py work_dirs/my_yolox/epoch_12.pth
```

# 结果生成
## 绘制分类损失曲线图
```
python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
```
## 绘制分类损失、回归损失曲线图，保存图片为对应的 pdf 文件
```
python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
```
## 在相同图像中比较两次运行结果的 bbox mAP
```
python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
```
## 计算平均训练速度
```
python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
```
## 配置与结果持久化存储
### 配置
```
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
```
### 持久化存储
```
python tools/test.py configs/yolox/my_yolox.py work_dirs/my_yolox/epoch_12.pth --show-dir imgs/
```
# 联系信息
如果你有任何问题或建议，请随时通过我的电子邮件（sibochen@yeah.net）与我联系。
