# VRDL_InstanceSegmentation

This is a project of implement instance segementation. We had implement the task base on following backbone modules:

1. Detectron2, from https://github.com/facebookresearch/detectron2 or https://github.com/conansherry/detectron2 for windows build

2. model weight link: https://drive.google.com/file/d/1XvZv_GsZNF_eIfrH7RvSGpfHDn5TT5SK/view?usp=sharing. After downloading, please put it in output file.


## Dataset Preprocessing

I use generate_coco_file.py to prepare the coco json format. It will output train/validation coco json file and classification of training/validation image

```
python generate_coco_file.py
```

## train model

I use detectron.py to train my model. It will save the model weight in output/ directory every 200 iterations. The final model weight is saved as "model_final.pth"   
```
python detectron.py
```
