import cv2
import random
import glob
import numpy as np
import pdb

def get_bg(input_img):
    # 定义结构元素大小
    kernel_size = (5, 5)
    # 创建自定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    bg_mask = cv2.inRange(img_gray, 100, 255)
    # 对图像进行腐蚀操作
    bg_mask = cv2.erode(bg_mask, kernel, iterations=1)
    fg_pos = np.where(bg_mask==0)
    new_img = input_img * cv2.merge([bg_mask, bg_mask ,bg_mask])
    img_h, img_w = new_img.shape[:2]
    
    for pos_h, pos_w in zip(fg_pos[0], fg_pos[1]):
        color_list = []
        ratio_list = [3, 5, 9, 15, 25, 35, 55]
        for ratio_item in ratio_list:
            for tmp_i in range(ratio_item):
                for tmp_j in range(ratio_item):
                    tmp_h = pos_h + (tmp_i - int(ratio_item/2))
                    tmp_w = pos_w + (tmp_j - int(ratio_item/2))
                    if tmp_h < 0 or tmp_w < 0 or tmp_h >= img_h or tmp_w >= img_w:
                        continue
                    if bg_mask[tmp_h, tmp_w] != 0:
                        # print(pos_h, pos_w, tmp_h, tmp_w, ratio_item)
                        # print(input_img[pos_h, pos_w])
                        color_list.append(input_img[tmp_h, tmp_w, ::-1])
            if len(color_list) > 2:
                break
        # print(color_list, np.mean(np.array(color_list), axis=0).astype(np.uint8))
        # new_img[pos_h, pos_w] = np.mean(np.array(color_list), axis=0).astype(np.uint8)
        new_img[pos_h, pos_w] = random.choice(color_list)
    # new_img = cv2.GaussianBlur(new_img, (25,25), 0, 1)
    cv2.imshow('test', new_img[:,:,::-1])
    cv2.waitKey()
    
def main():
    img_files = glob.glob('../char_detection/data/images/test.jpeg')
    for img_file in img_files:
        get_bg(cv2.imread(img_file))
        break

main()



 
