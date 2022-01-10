import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.models.common import Conv
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = attempt_load("yolov5s.pt", map_location=device)

stride = int(model.stride.max())

names = model.module.names if hasattr(model, 'module') else model.names
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000
agnostic_nms = False
print(names)

cap = cv2.VideoCapture(0)
prev_frame_time = 0

new_frame_time = 0
cam = cv2.VideoCapture("C:\\Users\\admin\\Pictures\\New folder\\YoloV5-and-DeepSort-Custom-Dataset-main\\video_08.avi")
while True:
    ret_val, img = cam.read()
    #img = cv2.resize(img,(640,640))
    img2 = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img2 = np.ascontiguousarray(img2)
    img2 = torch.from_numpy(img2)
    img2 = img2.unsqueeze(0)
    img2 = img2.to(device)
    img2 = img2/255.0
    pred = model(img2)[0]
    pred = pred.cpu()

    preds = non_max_suppression(pred, conf_thres, iou_thres,None, agnostic_nms, max_det=max_det)
    preds = preds[0]

    preds = preds.numpy()
    preds = preds.tolist()
    for pred in preds:
        cv2.rectangle(img,(int(pred[0]),int(pred[1])),(int(pred[2]),int(pred[3])),color=(0,0,255))
        cv2.putText(img,names[int(pred[5])], (int(pred[0]),int(pred[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(img, fps, (7, 70),cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('yolo v5', img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
