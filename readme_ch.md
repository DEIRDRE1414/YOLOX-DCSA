# Prepare dataset
```
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
COCO Annotation format
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
The experimental data set is collected from Google Earth and some open source data sets, and remote sensing images containing aircraft categories are obtained to construct a remote sensing data set with aircraft as the target. Aircraft targets in the image were annotated using the image annotation tool, bounding boxes were drawn for each target, and their category and location information were recorded. The constructed dataset contains 5 174 images for training and 2 301 images for validation. The dataset was organized in COCO format. Due to the large size of the dataset, it is not uploaded on github. Please contact the author by email if you have any requirements.
# Prepare the configuration file

```
## From '_base_ = '.. /yolox/yolox_s_8xb8-300e_coco.py' 'to build the basic structure of the model.
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
# Model training
##  Single GPU training
```
python tools/train.py /configs/yolox/my_yolox.py
```

## CPU training
```
 export CUDA_VISIBLE_DEVICES=-1
 python tools/train.py /configs/yolox/my_yolox.py
```

# Model validation
## Single GPU validation
```
python tools/test.py configs/yolox/my_yolox.py work_dirs/my_yolox/epoch_12.pth
```
## CPU validation
```
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py configs/yolox/my_yolox.py work_dirs/my_yolox/epoch_12.pth
```

# Generate results
## Plot the classification loss
```
python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
```
## Plot the classification loss/regression loss and save them as pdf files
```
python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
```
## Bbox mAP comparing the results of two runs on the same image
```
python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
```
## Calculate the average training speed
```
python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
```
## Configuration and result persistence
### Configuration
```
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
```
### Persistent Storage
```
python tools/test.py configs/yolox/my_yolox.py work_dirs/my_yolox/epoch_12.pth --show-dir imgs/
```
#  Contact information
If you have any questions or suggestions, please feel free to reach me at my email (sibochen@yeah.net).
