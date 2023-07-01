
# YOLOV8 with DeepSort(Pytorch) Object-Tracking

## Introduction

This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT) implimentaion in Pytorch and intergrated with YOLOV8. 
## Dependencies

The code was implemented with Python 3.10 in mind. can be implemented in other python until the particular requirements is fulfilled. The following dependencies are needed to run the tracker:


* easydict
* numpy
* opencv-python
* torch
* torchvision
* ultralytics
* gdown

Inorder to enable Gpu for the detection kindly install the torch version based on your gpu version from 
* https://pytorch.org/get-started/locally/

## Installation

See below for a quickstart installation 


Create a virtual environment

```
python -m venv venv
```

```
venv\scripts\activate
```

Install the requirements
 
```
pip install -r requirements.txt
```

Downloading weights to the corresponding folder 

```
cd deep_sort_pytorch\deep_sort\deep\checkpoint
```

```
gdown "https://drive.google.com/uc?export=download&id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN"
```

```
cd ..\..\..\..\
```

```
python main.py
```





## Reference

 - [Deepsort Pytorch](https://github.com/ZQPei/deep_sort_pytorch)

 - [Original Deepsort](https://github.com/nwojke/deep_sort)

 - [YOLO V8](https://github.com/ultralytics/ultralytics)

## License

[MIT](https://choosealicense.com/licenses/mit/)

