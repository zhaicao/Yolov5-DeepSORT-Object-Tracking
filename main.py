from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QPushButton, QTableWidgetItem, QMainWindow, QHeaderView
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Slot

import sys
import cv2
import numpy as np
import torch

from utils.util import readConfigs, initFlowMetrix
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size

# 在QApplication之前先实例化
from logic.service.imageDetector import ImgDetector
from logic.service.videoDetector import VideoDetector

uiLoader = QUiLoader()

class ObjectTracking(QMainWindow):

    def __init__(self):
        super().__init__()
        # 参数配置
        self.image_size = 500
        self.video_size = 700
        # 是否跳出当前循环的线程
        self.jump_threading: bool = False
        # 初始设置为摄像头
        self.vid_source = '0'
        # 检测视频的线程
        self.track_thread = False

        self.device = ''
        self.confidence = 0.35
        self.iou_threshold = 0.45
        self.conf = readConfigs()
        # 流量list
        self.flowTabList = initFlowMetrix(self.conf["yolo"]["detect_obj"])

        # 指明模型加载的位置的设备
        self.model = self.model_load(weights=self.conf["yolo"]["weight"],
                                     device=self.device)

        # 加载界面
        self.ui = uiLoader.load('ui/main.ui')
        # 界面初始化
        self.initWindow()

        # 连接槽函数
        self.ui.imgUploadBtn.clicked.connect(self.handleImgUpload)
        self.ui.imgDetectBtn.clicked.connect(self.handleImgDetect)

        self.ui.openCameraDetectBtn.clicked.connect(self.handleOpenCameraDetect)
        self.ui.videoUpdateBtn.clicked.connect(self.handleVideoUpdate)
        self.ui.videoDetectBtn.clicked.connect(self.handleVideoDetect)

    def handleImgUpload(self):
        """
        上传图片
        Returns:
        """
        # 选择图片进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self.ui, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            self.ui.img2predict = fileName
            # 进行左侧原图展示
            self.img = cv2.imread(fileName)
            # 调整一下图片的大小
            img = self.resize_img(self.img, self.image_size)
            img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
            self.ui.leftImgLab.setPixmap(QPixmap.fromImage(img))
            self.ui.leftImgLab.setAlignment(Qt.AlignCenter)
            # 上传图片之后右侧的图片重置
            self.ui.rightImgLab.clear()
            self.enable_btn(self.ui.imgDetectBtn)

    def handleImgDetect(self):
        """
        图像检测
        Returns:

        """
        # 对单张图片进行检测
        detector = ImgDetector()
        img_detected = detector.imgDetect(self.img)
        img = self.resize_img(img_detected, self.image_size)
        img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
        # 清除上一次检测的结果
        self.ui.rightImgLab.clear()
        self.ui.rightImgLab.setPixmap(QPixmap.fromImage(img))

    def handleOpenCameraDetect(self):
        """
        摄像头检测
        视频和摄像头的主函数是一样的，传入的source不同
        """
        self.vid_source = '0'
        if not self.track_thread or not self.track_thread.isRunning():
            self.track_thread = VideoDetector(self.model, self.video_size, self.device, self.conf,
                                              self.disable_btn, self.stride, self.ui, self.flowTabList,
                                              self.resize_img, self.updateFlowTab, self.vid_source,
                                              self.ui.leftVideoLab, self.ui.rightVideoLab)

            self.track_thread.track_status.connect(self.updateStatus)
            self.track_thread.started.connect(self.updateButtonState)  # 开始后更新Btn
            self.track_thread.finished.connect(self.updateButtonState)  # 结束后更新Btn

            self.track_thread.start()  # 线程开始
            self.updateStatus("开始检测")
        else:
            self.track_thread.stopDetect()
            self.track_thread.wait()
            self.updateStatus("结束检测")


    def handleVideoUpdate(self):
        """
        上传视频文件
        """
        print("上传视频文件")
        fileName, fileType = QFileDialog.getOpenFileName(self.ui, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.disable_btn(self.ui.openCameraDetectBtn)
            self.disable_btn(self.ui.videoUpdateBtn)
            self.enable_btn(self.ui.videoDetectBtn)
            # 生成读取视频对象
            cap = cv2.VideoCapture(fileName)
            # 获取视频的帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 显示原始视频帧率
            self.ui.raw_fps_value.setText(str(fps))
            if cap.isOpened():
                # 读取一帧用来提前左侧展示
                ret, raw_img = cap.read()
                cap.release()
            else:
                QMessageBox.warning(self, "需要重新上传", "请重新选择视频文件")
                self.disable_btn(self.ui.videoDetectBtn)
                self.enable_btn(self.ui.openCameraDetectBtn)
                self.enable_btn(self.ui.videoUpdateBtn)
                return
            # 调整一下图片的大小
            img = self.resize_img(np.array(raw_img), self.video_size)
            img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
            self.ui.leftVideoLab.setPixmap(QPixmap.fromImage(img))
            # 上传图片之后右侧的图片重置
            self.ui.rightVideoLab.clear()
            self.vid_source = fileName
            self.jump_threading = False

    def handleVideoDetect(self):

        """
        视频检测
        """
        if not self.track_thread or not self.track_thread.isRunning():
            self.track_thread = VideoDetector(self.model, self.video_size, self.device, self.conf,
                                     self.disable_btn, self.stride, self.ui, self.flowTabList,
                                     self.resize_img, self.updateFlowTab, self.vid_source,
                                    self.ui.leftVideoLab, self.ui.rightVideoLab)

            self.track_thread.track_status.connect(self.updateStatus)
            self.track_thread.started.connect(self.updateButtonState)  # 开始后更新Btn
            self.track_thread.finished.connect(self.updateButtonState)  # 结束后更新Btn

            self.track_thread.start()  # 线程开始
            self.updateStatus("开始检测")
        else:
            self.track_thread.stopDetect()
            self.track_thread.wait()
            self.updateStatus("结束检测")

    @Slot(str)
    def updateStatus(self, text):
        print(text)

    def updateButtonState(self):
        """
           检测启动或者停止
        """
        if not self.track_thread or not self.track_thread.isRunning():
            self.enable_btn(self.ui.openCameraDetectBtn)
            self.enable_btn(self.ui.videoUpdateBtn)
            self.disable_btn(self.ui.videoDetectBtn)
            self.ui.videoDetectBtn.setText("开始检测")
        else:
            # 停止当前线程
            self.disable_btn(self.ui.openCameraDetectBtn)
            self.disable_btn(self.ui.videoUpdateBtn)
            self.enable_btn(self.ui.videoDetectBtn)
            self.ui.videoDetectBtn.setText("停止检测")

    def resize_img(self, img, size):
        """
        调整图片大小，方便用来显示
        @param img: 需要调整的图片
        @param size: 需要调整的尺寸
        """
        resize_scale = min(size / img.shape[0], size / img.shape[1])
        img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def disable_btn(self, pushButton: QPushButton):
        pushButton.setDisabled(True)

    def enable_btn(self, pushButton: QPushButton):
        pushButton.setEnabled(True)

    def initWindow(self):
        """
        界面重置
        """
        # self.ui.resize(1588, 1120)
        self.disable_btn(self.ui.imgDetectBtn)
        self.disable_btn(self.ui.videoDetectBtn)
        self.jump_threading = False

        # 初始化流量表格
        self.initFlowTab()

    @torch.no_grad()
    def model_load(self,
                   weights="",  # entities.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   ):
        """
        模型初始化
        """
        device = self.device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load entities
        model = attempt_load(weights, device)  # load FP32 entities
        self.stride = int(model.stride.max())  # entities stride
        self.video_size = check_img_size(self.video_size, s=self.stride)  # check img_size
        if half:
            model.half()  # to FP16
        # Run inference
        if device.type != 'cpu':
            print("Run inference")
            model(torch.zeros(1, 3, self.video_size, self.video_size).to(device).type_as(
                next(model.parameters())))  # run once
        print("模型加载完成!")
        return model

    def closeEvent(self, event):
        """
        界面关闭事件
        """
        reply = QMessageBox.question(
            self.ui,
            '退出',
            "确定退出?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.jump_threading = True
            self.ui.close()
            event.accept()
        else:
            event.ignore()

    def initFlowTab(self):
        """
        初始化流量表格
        """
        # 首列固定，其他列自适应
        rows, cols = self.flowTabList.shape
        self.ui.flowTab.setRowCount(rows - 1)
        self.ui.flowTab.setColumnCount(cols)
        self.ui.flowTab.setHorizontalHeaderLabels(self.flowTabList[0, :].tolist())
        self.ui.flowTab.horizontalHeader().resizeSection(0, 100)
        self.ui.flowTab.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        for column in range(1, self.flowTabList.shape[1] + 1):
            self.ui.flowTab.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

        self.updateFlowTab()

    def updateFlowTab(self):
        """
        更新检测流量表格
        """
        for index, elem in np.ndenumerate(self.flowTabList):
            if index[0] != 0:  # 排除第一行
                item = QTableWidgetItem(elem)
                item.setTextAlignment(Qt.AlignCenter)
                self.ui.flowTab.setItem(index[0] - 1, index[1], item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ot = ObjectTracking()
    ot.ui.show()
    app.exec()