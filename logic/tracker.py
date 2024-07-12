"""
封装的deepsort跟踪器脚本
"""
import cv2
import torch
import numpy as np
from collections import deque

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class Tracker(object):
    def __init__(self, conf):
        cfg = get_config()
        cfg.merge_from_file(conf["deepsort"]["config_file"])
        self.deepsort = DeepSort(conf["deepsort"]["reid_ckpt"],
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.data_deque = {}


    def plot_bboxes(self, image, bboxes, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            # 设置锚框颜色
            if cls_id in ['person']:
                color = (85, 45, 255)
            elif cls_id in ['motorcycle']:
                color = (170, 178, 30)
            elif cls_id in ["car"]:
                color = (222, 82, 175)
            elif cls_id in ["bus"]:
                color = (0, 149, 255)
            else:
                color = (255, 0, 0)
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=int(tl * 0.7), lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            label = '{}:{}'.format(pos_id, cls_id)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] - 20, c1[1] - t_size[1] + 6
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 4,
                        [225, 255, 255], thickness=int(tf / 2), lineType=cv2.LINE_AA)
        return image


    def update_tracker(self, target_detector, image):
        # 返回的boxes里面包含了检测到的目标的个数，以及每一个目标的目标框左上角以及右下角坐标，class类别以及置信度
        _, bboxes = target_detector.detect(image)
        bbox_xywh = []
        confs = []
        bboxes2draw = []
        clss = []
        if len(bboxes):
            # Adapt detections to deep sort input format 将所有检测结果转换成deepsort格式
            for x1, y1, x2, y2, label, conf in bboxes:
                # deepsort输入的是中心点坐标以及宽高
                obj = [
                    int((x1 + x2) / 2), int((y1 + y2) / 2),
                    x2 - x1, y2 - y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)
                clss.append(label)
                # torch.Tensor()传入的是一个列表
            xywhs = torch.Tensor(bbox_xywh)  # bbox_xywh检测框的中心点坐标以及宽高
            confss = torch.Tensor(confs)  # 置信度

            # Pass detections to deepsort：将检测结果传递给deepsort
            outputs = self.deepsort.update(xywhs, confss, clss, image)
            if len(outputs) != 0:
                # 删除已经不存在track_id
                object_list = np.array(outputs)[:, -1].transpose().tolist()
                for key in list(self.data_deque):
                    if str(key) not in object_list:
                        self.data_deque.pop(key)

            for value in list(outputs):
                # 左上角坐标，右下角坐标以及跟踪id
                x1, y1, x2, y2, label, track_id = value
                # 计算中心点
                center_x = int((x1 + x2) * 0.5)
                center_y = int((y1 + y2) * 0.5)
                bboxes2draw.append(
                    (x1, y1, x2, y2, label, track_id)
                )
                center = (center_x, center_y)
                """
                轨迹数据
                """
                # trackid不在队列中，则增加deque
                if track_id not in self.data_deque:
                    self.data_deque[track_id] = deque(maxlen=64)
                # 插入轨迹数据
                self.data_deque[track_id].appendleft(center)
                # 轨迹颜色
                if label == "person":
                    color = (85, 45, 255)
                elif label == "motorcycle":
                    color = (170, 178, 30)
                elif label == "car":
                    color = (222, 82, 175)
                elif label == "bus":
                    color = (0, 149, 255)
                else:
                    color = (255, 0, 0)
                # 绘制轨迹
                for i in range(1, len(self.data_deque[track_id])):
                    # 轨迹值是否存在
                    if self.data_deque[track_id][i - 1] is None or self.data_deque[track_id][i] is None:
                        continue
                    # 动态轨迹粗细
                    thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                    # 绘制轨迹
                    cv2.line(image, self.data_deque[track_id][i - 1], self.data_deque[track_id][i], color, thickness)
        # 绘制box
        image = self.plot_bboxes(image, bboxes2draw)
        return image, bboxes2draw
