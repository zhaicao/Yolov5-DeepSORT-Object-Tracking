# Yolov5-DeepSORT-Object-Tracking
This is a PySide6 GUI program that is based on YOLOv5 and DeepSORT for traffic object detection and tracking. The program records the speed and traffic volume of objects and displays a warning message when an object is speeding.

**Features:**
* Supports video, camera, and image detection
* Shows categories, speeds, and trails of customized objects
* Records traffic volume of customized objects
* Supports video start and stop detection
* Outputs records of speeding

## Installation
1. Python 3.10 or later with all requirements.txt dependencies installed, including torch>=2.1.2 To install run:
```bash
pip install -r requirements.txt
```
2. Download or train a model and put the weight file into the `weights` folder. You can download the YOLOv5 weight from [Releases · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/releases) or directly download [yolov5s.pt](https://drive.google.com/file/d/1GboqAYsnlnf4_XNm2Uy9uK5mnLcu21l1/view?usp=drive_link)
3. Download the Deep-SORT weight file [ckpt.t7](https://drive.google.com/file/d/1GcJciXMqUss4PW8tFl18vb0ctE7yjmz7/view?usp=drive_link) and put it into the `/deep_sort/deep_sort/deep/checkpoint`folder.
4. Download the MobileNet [weight](https://drive.google.com/file/d/1A35lVW_TKQZKXRGImCjZE3gqfT9b1JDm/view?usp=drive_link) and [prototxt](https://drive.google.com/file/d/1sznXIHi1PEj3H94Xqb0j008HRrRaCLS2/view?usp=drive_link) file for image detection or choose Yolo v5 as the detector. Note that the latter choice will require modifying the code.

**Tips:**
- You can modify the `configs.yaml` file to change the default settings, such as the weight paths.

## Run

```bash
python main.py
```

# Result
#### Vehicles Detection, Tracking and Counting 
![](./imgs/demo.png)

# References
https://github.com/ultralytics/yolov5