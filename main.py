from facedetect.facedetect import *
from mosaic import *
import os
import argparse
import time
import cv2
from cv2 import dnn
import numpy as np




def mosaicFaces():
    net = dnn.readNetFromONNX(args.onnx_path)  # onnx version
    input_size = [int(v.strip()) for v in args.input_size.split(",")]
    witdh = input_size[0]
    height = input_size[1]
    priors = define_img_size(input_size)
    result_path = args.results_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    listdir = os.listdir(facePath)
    for file_path in listdir:
        img_path = os.path.join(facePath, file_path)
        img_ori = cv2.imread(img_path)
        rect = cv2.resize(img_ori, (witdh, height))
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        net.setInput(dnn.blobFromImage(rect, 1 / image_std, (witdh, height), 127))
        time_time = time.time()
        boxes, scores = net.forward(["boxes", "scores"])
        print("inference time: {} s".format(round(time.time() - time_time, 4)))
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)           # 将 boxes 坐标转换为图像坐标
        boxes = center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(img_ori.shape[1], img_ori.shape[0], scores, boxes, args.threshold)   # boxes, 框的位置 shape = (8,4)， labels 标签， probes 人脸置信度

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            mosaic(img_ori, (box[0], box[1]), (box[2]-box[0], box[3]-box[1]), args.grad_size)
            if(args.show_boxes):
                cv2.rectangle(img_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(result_path, 'boxes_' + file_path), img_ori)
        print("result_pic is written to {}".format(os.path.join(result_path, file_path)))
        cv2.imshow("ultra_face_ace_opencvdnn_py", img_ori)
        cv2.waitKey(-1)
    cv2.destroyAllWindows()



parser = argparse.ArgumentParser()
parser.add_argument('--onnx_path', default="./facedetect/version-RFB-320_simplified.onnx", type=str, help='onnx version')       # dnn weights
parser.add_argument('--input_size', default="320,240", type=str, help='define network input size,format: width,height')
parser.add_argument('--threshold', default=0.5, type=float, help='face score threshold')                                        # face detect threshold
parser.add_argument('--imgs_folder', default="./imgs", type=str, help='imgs dir')
parser.add_argument('--results_path', default="results", type=str, help='results dir')
parser.add_argument('--grad_size', default=16, type=int, help ='size of mosaic grid')
parser.add_argument('--show_boxes', default=False, type=bool, help = 'display face location box or not')
args = parser.parse_args()

facePath = args.imgs_folder


if __name__ == '__main__':
    mosaicFaces()