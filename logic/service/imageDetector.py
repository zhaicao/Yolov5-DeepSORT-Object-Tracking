import cv2
import cvzone
import numpy as np
import copy

class ImgDetector:
    def __init__(self):
        # 设置图片的宽度和高度
        self.img_width, self.img_heigth = 300, 300
        # 得到图像的高宽比
        self.WHRatio = self.img_width / float(self.img_heigth)
        # 设置图片的缩放因子
        self.ScaleFactor = 0.007843
        # 设置平均数
        self.meanVal = 127.5
        # 设置置信度阈值
        self.threshod = 0.4

        # mobileNetSSD可以检测类别数21=20+1（背景）
        self.classNames = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse',
                      'motorbike', 'person', 'pottedplant',
                      'sheep', 'sofa', 'train', 'tvmonitor']

        # 加载模型
        self.net = cv2.dnn.readNetFromCaffe(prototxt='weights/deploy.prototxt',
                                            caffeModel='weights/mobilenet_iter_73000.caffemodel')

    # 对图片进行处理和设置网络的输入同时进行前向传播
    def imgDetect(self, img_source):
        # 浅拷贝，否则对象被覆盖
        img_detected = copy.copy(img_source)
        # 对图片进行预处理
        # 将原始图像转换为适合输入深度学习模型的标准化、尺寸调整和通道重排后的数据格式
        blob = cv2.dnn.blobFromImage(image=img_detected,
                                     scalefactor=self.ScaleFactor,
                                     size=(self.img_width, self.img_heigth),
                                     mean=self.meanVal)
        # 设置网络的输入并进行前向传播
        self.net.setInput(blob)
        detections = self.net.forward()
        height, width, channel = np.shape(img_detected)

        # 遍历检测的目标
        for i in range(detections.shape[2]):
            # 置信度
            confidence = round(detections[0, 0, i, 2] * 100, 2)
            if confidence > self.threshod:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * width)
                yLeftBottom = int(detections[0, 0, i, 4] * height)
                xRightTop = int(detections[0, 0, i, 5] * width)
                yRightTop = int(detections[0, 0, i, 6] * height)

                cv2.rectangle(img=img_detected, pt1=(xLeftBottom, yLeftBottom),
                              pt2=(xRightTop, yRightTop), color=(0, 255, 0), thickness=2)
                label = self.classNames[class_id] + ": " + str(confidence)

                cvzone.putTextRect(img=img_detected, text=label, pos=(xLeftBottom + 9, yLeftBottom - 12),
                                   scale=1, thickness=1, colorR=(255, 0, 0))
        return img_detected