'''
    author: soultop
    created: 2019-9-10
    code: utf-8
'''

import cv2

def mosaic(frame, stPoint, size, neighbor=8):
    '''
    :param frame:       input image,  shape = [:,:,3]
    :param stPoint:     moscia box left_top point axis
    :param size:        moscia siize
    :param neighbor:    grid_size
    :return:
    '''

    x, y = stPoint
    w, h = size

    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return

    for i in range(0, h, neighbor):
        for j in range(0, w, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            h = frame[i+y:i+y+neighbor, j+x:j+x+neighbor]

            ls = []
            for t in range(h.shape[2]):
                ls.append(h[:, :, t].mean(dtype=int).tolist())

            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)
            cv2.rectangle(frame, left_up, right_down, ls, -1)





'''
test for mosaic function
'''

import argparse
import os

parser = argparse.ArgumentParser(description='Creates a photomosaic from input images')
parser.add_argument('--input-folder', default='./imgs/')
parser.add_argument('--grid-size', default=8)
parser.add_argument('--output-folder', default='./results/', required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    output_file = args.output_folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    input_list = os.listdir(args.input_folder)

    for file in input_list:
        im = cv2.imread(args.input_folder + file)
        mosaic(im, (0, 0), (im.shape[1], im.shape[0]), args.grid_size)
        cv2.imwrite(output_file + os.path.splitext(file)[0] + '_mosaic' + str(args.grid_size) + os.path.splitext(file)[1], im)