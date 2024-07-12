import time
import cv2
import random
import numpy as np
import torch.backends.cudnn as cudnn

from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal, Slot

from logic.tracker import Tracker
from logic.detector import Detector
from deep_sort.tools.highway_detection import HightwayTracker
from logic.highwayAnalyse import HighwayAnalyse
from utils.datasets import LoadImages, LoadStreams
from utils.util import updateFlowMetrix

class VideoDetector(QThread):
    """
    摄像头/视频检测
    """

    track_status = Signal(str)

    def __init__(self, model, video_size, device, conf,
                 disable_btn, stride, widget, flowTabList,
                 resize_img, updateFlowTab, source, leftVideoLab, rightVideoLab):
        super(VideoDetector, self).__init__()
        self.model = model
        self.video_size = video_size
        self.device = device
        self.conf = conf
        self.disable_btn = disable_btn
        self.stride = stride
        self.widget = widget
        self.flowTabList = flowTabList
        self.source = source
        self.resize_img = resize_img
        self.updateFlowTab = updateFlowTab

        self.left_img = leftVideoLab
        self.right_img = rightVideoLab

        self.stopped = False

    @Slot(str)
    def run(self):
        """
        视频检测thread
        """
        # 原始视频大小
        width = 1920
        height = 1080
        # 缩放比例
        resize_scale = 0.7

        model = self.model
        img_size = [self.video_size, self.video_size]  # inference size (pixels)
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu

        # 实例化yolov5检测器
        detector = Detector(self.conf)
        # 实例化追踪器和速度追踪器
        tracker = Tracker(self.conf)
        tracker_hightway = HightwayTracker(self.conf)
        # 实例化分析类
        highway = HighwayAnalyse()

        f = open("log/OverSpeed.txt", 'a')
        f.write("------------------------------------------------------------------------")
        f.write("\n")
        f.write("开始记录，记录时间为{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        f.write("\n")
        f.close()

        half = device.type != 'cpu'  # half precision only supported on CUDA
        # 判断数据源
        if self.source == "":
            # self.disable_btn(self.det_img_button)
            QMessageBox.warning(self.widget, "请上传", "请先上传视频或图片再进行检测")
        else:
            source = str(self.source)
            webcam = source.isnumeric()
            # 设置数据源 摄像头/视频文件
            if webcam:
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=img_size, stride=self.stride)
            else:
                dataset = LoadImages(source, img_size=img_size, stride=self.stride)
            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # 填充第一个撞线polygon（蓝色）
            mask_image_temp = np.zeros((height, width), dtype=np.uint8)
            list_pts_blue = [[0, 450], [1920, 450], [1920, 460], [0, 460]]
            ndarray_pts_blue = np.array(list_pts_blue, np.int32)
            # cv2.fillPoly 填充多边形
            polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
            # np.newaxis增加一个维度，变成三维图像
            polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

            # 填充第二个撞线polygon（黄色）
            mask_image_temp = np.zeros((height, width), dtype=np.uint8)

            list_pts_red = [[0, 550], [1920, 550], [1920, 560], [0, 560]]
            ndarray_pts_red = np.array(list_pts_red, np.int32)
            polygon_red_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_red], color=2)
            polygon_red_value_2 = polygon_red_value_2[:, :, np.newaxis]

            # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
            polygon_mask_blue_and_red = polygon_blue_value_1 + polygon_red_value_2

            # 缩小尺寸
            polygon_mask_blue_and_red = cv2.resize(polygon_mask_blue_and_red,
                                                   (int(width * resize_scale), int(height * resize_scale)))

            # 蓝 色盘 b,g,r
            blue_color_plate = [255, 0, 0]
            # 蓝 polygon图片
            blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

            # 红 色盘
            red_color_plate = [0, 0, 255]
            # 红 polygon图片
            red_image = np.array(polygon_red_value_2 * red_color_plate, np.uint8)
            # 彩色图片（值范围 0-255）
            color_polygons_image = blue_image + red_image
            # 缩小尺寸
            color_polygons_image = cv2.resize(color_polygons_image,
                                              (int(width * resize_scale), int(height * resize_scale)))
            # list 与蓝色polygon重叠
            list_overlapping_blue_polygon = []

            # list 与黄色polygon重叠
            list_overlapping_red_polygon = []

            # 计算帧率开始时间
            frame_index = -2

            for path, img, im0s, vid_cap in dataset:
                yolo_bboxs_l = []

                # 直接跳出for，结束线程
                if self.stopped:
                    break
                # 打开视频
                # 读取每帧图片
                fps_start_time = time.time()

                if source == '0': im0s = im0s[0]
                im = cv2.resize(im0s, (int(width * resize_scale), int(height * resize_scale)))

                # 更新跟踪器(deepsort)
                # objtracker.update返回的是image和bounding box
                output_image_frame, list_bboxs = tracker.update_tracker(detector, im)
                # 输出图片
                output_image_frame = cv2.add(output_image_frame, color_polygons_image)

                if len(list_bboxs) > 0:
                    # ----------------------判断撞线----------------------
                    for item_bbox in list_bboxs:
                        x1, y1, x2, y2, label, track_id = item_bbox
                        yolo_bboxs_l.append((x1, y1, x2, y2, label, track_id))
                        # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                        y1_offset = int(y1 + ((y2 - y1) * 0.6))
                        # 撞线的点
                        y = y1_offset
                        x = x1
                        if polygon_mask_blue_and_red[y, x] == 1:
                            # 如果撞 蓝polygon
                            if track_id not in list_overlapping_blue_polygon:
                                list_overlapping_blue_polygon.append(track_id)
                            # 判断 黄polygon list 里是否有此 track_id
                            # 有此 track_id，则 认为是 北向
                            if track_id in list_overlapping_red_polygon:
                                # 北向+1
                                self.flowTabList = updateFlowMetrix(self.flowTabList, label, "north", 1)
                                # 删除 黄polygon list 中的此id
                                list_overlapping_red_polygon.remove(track_id)

                        elif polygon_mask_blue_and_red[y, x] == 2:
                            # 如果撞 黄polygon
                            if track_id not in list_overlapping_red_polygon:
                                list_overlapping_red_polygon.append(track_id)
                            # 判断 蓝polygon list 里是否有此 track_id
                            # 有此 track_id，则 认为是 南向
                            if track_id in list_overlapping_blue_polygon:
                                # 南向+1
                                self.flowTabList = updateFlowMetrix(self.flowTabList, label, "south", 1)
                                list_overlapping_blue_polygon.remove(track_id)
                    sp = im.shape
                    # 更新速度追踪
                    tracker_bboxs_hw = tracker_hightway.update_tracker(output_image_frame, yolo_bboxs_l)
                    if (frame_index % 15) == 0:
                        highway.update_id_info(sp, tracker_bboxs_hw, 'left')
                    highway.plot_bboxes_1(output_image_frame, tracker_bboxs_hw, 'left')
                    frame_index += 1
                    # ----------------------清除无用id----------------------
                    list_overlapping_all = list_overlapping_red_polygon + list_overlapping_blue_polygon
                    for id1 in list_overlapping_all:
                        is_found = False
                        for _, _, _, _, _, bbox_id in list_bboxs:
                            if bbox_id == id1:
                                is_found = True
                                break
                        if not is_found:
                            # 如果没找到，删除id
                            if id1 in list_overlapping_red_polygon:
                                list_overlapping_red_polygon.remove(id1)
                            if id1 in list_overlapping_blue_polygon:
                                list_overlapping_blue_polygon.remove(id1)
                    list_overlapping_all.clear()
                    # 清空list
                    list_bboxs.clear()
                else:
                    # 如果图像中没有任何的bbox，则清空list
                    list_overlapping_blue_polygon.clear()
                    list_overlapping_red_polygon.clear()
                    pass
                pass

                fps = int(1 / (time.time() - fps_start_time))

                # 更新界面流量和fps数据
                self.updateFlowTab()
                self.widget.detect_fps_value.setText(str(fps))

                # cv2.imshow('im', im0s)

                img = self.resize_img(im0s, self.video_size)
                img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1],
                             QImage.Format_RGB888)
                self.left_img.setPixmap(QPixmap.fromImage(img))
                # cv2.imshow('output_image_frame', output_image_frame)
                img = self.resize_img(output_image_frame, self.video_size)
                img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1],
                             QImage.Format_RGB888)
                self.right_img.setPixmap(QPixmap.fromImage(img))

                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration
            # 使用完摄像头释放资源
            f = open("log/OverSpeed.txt", 'a')
            f.write("结束记录，记录时间为{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("\n")
            f.write("------------------------------------------------------------------------")
            f.write("\n")
            f.close()
            if webcam:
                dataset.release()
            else:
                dataset.cap and dataset.cap.release()

    def stopDetect(self):
        self.stopped = True