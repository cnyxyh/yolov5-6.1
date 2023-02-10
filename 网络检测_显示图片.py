import base64
import json

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
import win32con
import win32gui

from config import *
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

HIDE_CONF = False  # 隐藏置信度
HIDE_LABELS = False  # 隐藏标签


# 选择设备
def get_model():
    device = select_device('')  # device 设备
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(WEIGHTS, map_location=device)  # 模型路径 在配置文件config.py里
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    return model, device, half, stride, names


# 获取模型、步幅以及类别

model, device, half, stride, names = get_model()  # 调用 load_model import get_model
imgsz = check_img_size(IMGSZ, s=stride)  # check image size


# 图像识别
# 不进行梯度处理
@torch.no_grad()
def pred_img(img0):
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    # 归一化处理
    img = img / 255.0  # 0 - 255 to 0.0 - 1.
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    pred = model(img, augment=False, visualize=False)[0]
    # NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=1000)
    det = pred[0]
    im0 = img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(names))
    # xywh_list = []
    lsxx = {"PaddleOCR": []}
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            lsxx["PaddleOCR"].append({"ttxt": names[c], "score": f'{conf:.2f}',"rect": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]})
            # xywh_list.append({"ttxt": names[c], "rect": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],"score": f'{conf:.2f}'})
            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            # xywh_list.append(xywh)
            label = None if HIDE_LABELS else (names[c] if HIDE_CONF else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()  # 最终预测出来的图像数组
    return im0, json.dumps(lsxx)  # 返回图像 坐标数组
    # return im0, xywh_list  # 返回图像 坐标数组


# post网络
class S(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()  # 输出IP

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        # print("数据进入",post_data.decode('utf-8'))

        img, res = run_inference(post_data.decode('utf-8'))
        self.do_HEAD()
        self.wfile.write("{}".format(res).encode('utf-8'))  # 这里发送了 res
        xianshi(img)


# post网络调用
def run_inference(jpeg):
    # print('jpeg',len(jpeg))
    imgs = jpeg  # 这里调用了 be64的图片数据
    imgs = base64.b64decode(imgs)  # 解码
    img = cv2.imdecode(np.frombuffer(imgs, np.uint8), cv2.IMREAD_COLOR)  # 二进制数据流转np.ndarray [np.uint8: 8位像素]
    dict_new = pred_img(img)  # 得到img和坐标结果
    # print('打印结果', dict_new)
    return dict_new


def xianshi(img):
    while True:
        cv2.imshow('text', img)  # 展示窗口-- 参数  (窗口名,图片)
        k = cv2.waitKey(1)
        hwnd = win32gui.FindWindow(None, 'text')
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)
        if k % 256 == 27:  # ESC
            cv2.destroyAllWindows()  # 把所有窗口全结束
            exit('退出显示 ...')


if __name__ == '__main__':
    httpserver = ThreadingHTTPServer(("", 8001), S)
    # print("serverport:", str(httpserver.server_port))
    httpserver.serve_forever()
    # run()
    # input()
