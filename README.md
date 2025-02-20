# Optimizing Object Detection with Multispectral RGB/IR Fusion

## Intro
Official Code for my Master's thesis.

Multispectral Object Detection with multi-scale attention mechanism transformer and Yolov5.

## Abstract
Multispectral image pairs can provide the combined information, making object detection applications more reliable and robust in the open world. 
To fully exploit the different modalities, we present a simple yet effective cross-modality feature fusion approach, named Cross-Modality Fusion Transformer (CFT) in this paper. 
Unlike prior CNNs-based works, guided by the Transformer scheme, our network learns long-range dependencies and integrates global contextual information in the feature extraction stage. 
More importantly, by leveraging the self attention of the Transformer, the network can naturally carry out simultaneous intra-modality and inter-modality fusion, and robustly capture the latent interactions between RGB and Thermal domains, thereby significantly improving the performance of multispectral object detection. 
Extensive experiments and ablation studies on multiple datasets demonstrate that our approach is effective and achieves state-of-the-art detection performance. 
 
### Overview
<div align="left">
<img src="https://github.com/DocF/multispectral-object-detection/blob/main/cft.png" width="800">
</div>

## Installation 
Python>=3.6.0 is required with all requirements.txt installed including PyTorch>=1.7 (The same as yolov5 https://github.com/ultralytics/yolov5 ).

#### Clone the repo
    git clone https://github.com/sopacha/Fusion_Transformer.git
  
#### Install requirements
 ```bash
$ cd  Fusion_Transformer
$ pip install -r requirements.txt
```

## Dataset
-[FLIR]  [[Google Drive]](http://shorturl.at/ahAY4) [[Baidu Drive]](https://pan.baidu.com/s/1z2GHVD3WVlGsVzBR1ajSrQ?pwd=qwer) ```extraction code:qwer``` 

  A new aligned version.

-[LLVIP]  [download](https://github.com/bupt-ai-cz/LLVIP)

-[VEDAI]  [download](https://downloads.greyc.fr/vedai/)


## Dataset Structure
The dataset should be organized as follows:

```bash
├── dataset_name/
    ├── RGB/             # Folder containing RGB images
    │   ├── train/
    │   ├── val/
    │   ├── test/
    ├── IR/              # Folder containing Infrared (Thermal) images
    │   ├── train/
    │   ├── val/
    │   ├── test/
    ├── labels/          # Shared annotations for both modalities (YOLO format)
    │   ├── train/
    │   ├── val/
    │   ├── test/
```

Each folder (```train```, ```val```, ```test```) should contain corresponding RGB, IR, and label files. Each RGB image, IR image, and label file must share the same filename to ensure proper pairing.

You need to convert all annotations to YOLOv5 format.

Refer: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

## Run
#### Download the pretrained weights
yolov5 weights (pre-train) 
#### Change the data cfg
some example in data/multispectral/

#### Change the model cfg
some example in models/transformer/

note!!!   I used xxxx_transfomerx3_dataset_multi.yaml 

### Train Test and Detect
train: ``` python3 train.py --cfg ./models/transformer yolov5l_fusion_transformerx3_VEDAI_multi.yaml --data ./data/multispectral/VEDAI_sofi.yaml```

test: ``` python3 test.py --weights ./runs/train/xxxx/weights/best.pt --data ./data/multispectral/VEDAI_sofi_test.yaml```

#### References

https://github.com/ultralytics/yolov5
