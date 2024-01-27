import cv2
import random
import glob
import numpy as np
import pdb
import os
import uuid

import sys
sys.path.append("..")
from utils.utils import *

from tqdm import tqdm
from tqdm.contrib import tzip

def get_bg(input_img, method):
    # 定义结构元素大小
    kernel_size = (5, 5)
    # 创建自定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    if method == 'calligraphy':
        bg_mask = cv2.inRange(img_gray, 100, 255)
    elif method == 'stele':
        bg_mask = cv2.inRange(img_gray, 0, 100)
    # 对图像进行腐蚀操作
    bg_mask = cv2.erode(bg_mask, kernel, iterations=1)
    fg_pos = np.where(bg_mask!=-1)
    new_img = input_img * cv2.merge([bg_mask, bg_mask ,bg_mask])
    img_h, img_w = new_img.shape[:2]
    
    for pos_h, pos_w in tzip(fg_pos[0], fg_pos[1]):
        color_list = []
        ratio_list = [3, 5, 9, 15, 25, 35, 55, 75, 95, 133]
        for ratio_item in ratio_list:
            for tmp_i in range(ratio_item):
                for tmp_j in range(ratio_item):
                    tmp_h = pos_h + (tmp_i - int(ratio_item/2))
                    tmp_w = pos_w + (tmp_j - int(ratio_item/2))
                    if tmp_h < 0 or tmp_w < 0 or tmp_h >= img_h or tmp_w >= img_w:
                        continue
                    if bg_mask[tmp_h, tmp_w] != 0:
                        color_list.append(input_img[tmp_h, tmp_w, ::-1])
            if len(color_list) > 2:
                break
        new_img[pos_h, pos_w] = random.choice(color_list)
    # new_img = cv2.GaussianBlur(new_img, (25,25), 0, 1)
    return new_img[:,:,::-1]
    
def main(method):
    if method == 'stele':
        img_files = glob.glob('../char_detection/data/images/碑帖*.jpeg') + glob.glob('../char_detection/data/images/碑帖*.jpg')
    elif method == 'calligraphy':
        img_files = glob.glob('../char_detection/data/images/书法*.jpeg') + glob.glob('../char_detection/data/images/书法*.jpg')
    for img_file in img_files:
        img_name = os.path.basename(img_file)
        out_file = '../char_binary/img_bgs/' + img_name.replace('.jpeg', '_bg.png').replace('.jpg', '_bg.png')
        
        if os.path.exists(out_file):
            continue
        print(out_file)
        bg_img = get_bg(np_imread(img_file), method)
        cv2.imencode('.png', bg_img)[1].tofile(out_file)
        # cv2.imwrite(out_file, bg_img)
        '''
        while True:
            out_file = '../char_binary/img_bgs/' + str(uuid.uuid4()) + '.png'
            if not os.path.exists(out_file):
                cv2.imwrite(out_file, bg_img)
                break
        '''

main(method='stele')