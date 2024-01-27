import matplotlib
import cv2
import random
import cv2
from PIL import ImageFont, ImageDraw, Image
from fontTools.ttLib import TTFont
import glob
import numpy as np
import os
import json
import re
import pdb
import math
import uuid
import pdb


# bbox定义
# [x1,y1,x2,y2]
def fonttoUnicode(oneStr):
    t=oneStr
    if  t[:3] == 'uni':t=t.replace('uni','\\u') 
    if  t[:2] == 'uF':t=t.replace('uF','\\u') 
    return json.loads(f'"{t}"') 

def load_ttf(fontName):
    dic = {}
    font = TTFont(fontName, fontNumber=0)
    glyphNames = font.getGlyphNames()
    for i in glyphNames:
        try:
            if re.match('uni[A-Z0-9]+', i):
                dic[i] = fonttoUnicode(i) 
        except: 
            pass
    return dic

def draw_ttf(fontNames):
    for fontName in fontNames:
        text_dict = load_ttf(fontName)
        for key, val in text_dict.items():
            out_file = '/data1/dist1/porn/sitong5/word_imgs_zh/'+key+'.png'
            if os.path.exists(out_file):
                continue
            im = Image.new('1', (128, 128), 255)
            draw = ImageDraw.Draw(im) #修改图片
            pdb.set_trace()
            ft = ImageFont.truetype(
                font=fontName,
                size=100
            )
            ''''''
            word_width, word_height = draw.textsize(val, font=ft)
            draw.text(((128-word_width)/2,(128-word_height)/2), val, fill = (0), font=ft)
            # word_width, word_height = draw.textsize(val)
            # draw.text(((128-word_width)/2,(128-word_height)/2), val, fill = (0))

            im.save(out_file)

def draw_chinese_char(input_char, src_img, font, pos='center', font_color=None):
    # im = Image.new('1', (128, 128), 255)
    if src_img.shape[0] == 128 and src_img.shape[1] == 128:
        im_tmp = Image.fromarray(src_img)
    else:
        crop_size = random.randint(50, min(src_img.shape[:2]))
        crop_x = random.randint(0, src_img.shape[0]-crop_size)
        crop_y = random.randint(0, src_img.shape[1]-crop_size)
        im_tmp = src_img[crop_x:crop_x+crop_size,
                        crop_y:crop_y+crop_size]
        im_tmp = cv2.resize(im_tmp, (128, 128))
        im_tmp = Image.fromarray(im_tmp)

    im_gt = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(im_tmp)
    draw_gt = ImageDraw.Draw(im_gt) #修改图片
    
    if pos == 'random':
        font_size = 100 + random.randint(-10, 10)
        ft = ImageFont.truetype(font=font, size=font_size)
        word_width, word_height = draw.textsize(input_char, font=ft)
        lt_pos = ((128-word_width)/2 + random.randint(-10, 10), 
                  (128-word_height)/2 + random.randint(-10, 10))
        rb_pos = (lt_pos[0] + word_width, lt_pos[1] + word_height)
    else:
        assert pos == 'center'
        font_size = 100
        ft = ImageFont.truetype(font=font, size=font_size)
        word_width, word_height = draw.textsize(input_char, font=ft)
        lt_pos = [(128-word_width)/2, (128-word_height)/2]
        rb_pos = [lt_pos[0] + word_width, lt_pos[1] + word_height]
    if font_color is None:
        font_color = (random.randint(0,50), random.randint(0,50), random.randint(0,50))
    draw.text(lt_pos, input_char, fill = font_color, font=ft)
    draw_gt.text(lt_pos, input_char, fill = (0, 0, 0), font=ft)
    im_tmp = np.array(im_tmp)
    im_gt = np.array(im_gt)
    kernel_size = (random.randint(0,7)*2+1, random.randint(0,7)*2+1)
    cv2.GaussianBlur(im_tmp, kernel_size, 0, 1)
    return im_tmp, im_gt, [lt_pos[::-1], rb_pos[::-1]]

def draw_unicode(input_range):
    start_index = int(input_range[0], 16)
    end_index = int(input_range[1], 16) + 1
    cur_index = start_index
    while cur_index<=end_index:
        key = r'\u' + str(hex(cur_index))[2:]
        val = key.encode().decode("unicode_escape")
        out_file = '/data1/dist1/porn/sitong5/word_imgs_zh/'+key[2:]+'.png'
        if os.path.exists(out_file):
            continue
        im = Image.new('1', (128, 128), 255)
        draw = ImageDraw.Draw(im) #修改图片
            
        ft = ImageFont.truetype(
                font='../general/fonts/AlibabaPuHuiTi-2-55-Regular.ttf',
                size=100
            )

        word_width, word_height = draw.textsize(val, font=ft)
        draw.text(((128-word_width)/2,(128-word_height)/2), val, fill = (0), font=ft)
        im.save(out_file)
        cur_index += 1

def example():
    # 设置字体文件路径
    font_path = "simhei.ttf"
    
    # 创建画布和绘制对象
    image_width, image_height = (500, 400)
    canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8) + 255
    draw = ImageDraw.Draw(canvas)
    img = Image.fromarray(canvas)
    
    # 设置字体样式和位置
    text = "Hello World!"
    font_size = 60
    font = ImageFont.truetype(font_path, size=font_size)
    start_x, start_y = (int(image_width / 2 - len(text) * font_size / 2), int(image_height / 2))

    draw.text((start_x, start_y), text, fill=(0, 0, 0), font=font)  # 绘制文本
    result = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def draw_char(char, bg_img, font):
    pass

def isin_bbox(bbox_1, bbox_2, iou_thres=0.8):
    # is bbox 1 in bbox 2
    if isinstance(bbox_1[0], list):
        bbox_1 = bbox_1[0] + bbox_1[1]
    if isinstance(bbox_2[0], list):
        bbox_2 = bbox_2[0] + bbox_2[1]
    
    x1 = max(bbox_1[0], bbox_2[0])
    y1 = max(bbox_1[1], bbox_2[1])
    x2 = min(bbox_1[2], bbox_2[2])
    y2 = min(bbox_1[3], bbox_2[3])
    
    intersection = (x2 - x1) * (y2 - y1)
    area_box1 = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
    area_box2 = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])
    union = area_box1 + area_box2 - intersection
    
    bbox_iou_1 = intersection / area_box1 
    if bbox_iou_1 >= iou_thres:
        return True
    else:
        return False


def img_split(src_img, src_gt):
    img_h, img_w = src_img.shape[:2]
    out_img_list = []
    out_gt_list = []
    if img_h <= img_w:
        for i in range(int(img_w / img_h) + 1):
            img_bbox = [[0,i*img_h],[img_h,(i+1)*img_h]]
            if (i+1)*img_h > img_w:
                img_bbox = [[0,img_w-img_h], [img_h,img_w]]
            tmp_img = src_img[img_bbox[0][0]:img_bbox[1][0],
                              img_bbox[0][1]:img_bbox[1][1]]
            tmp_gt = []
            for gt_item in src_gt:
                if isin_bbox(gt_item, img_bbox):
                    bbox_item = [gt_item[0]-img_bbox[0][0], 
                                 gt_item[1]-img_bbox[0][1],
                                 gt_item[2]-img_bbox[0][0], 
                                 gt_item[3]-img_bbox[0][1]]
                    tmp_gt.append(bbox_item)
            out_img_list.append(tmp_img)
            out_gt_list.append(tmp_gt)
    if img_h > img_w:
        for i in range(int(img_h / img_w) + 1):
            img_bbox = [[i*img_w,0], [(i+1)*img_w,img_w]]
            if (i+1)*img_w > img_h:
                img_bbox = [[img_h-img_w,0], [img_h,img_w]]
            tmp_img = src_img[img_bbox[0][0]:img_bbox[1][0],
                              img_bbox[0][1]:img_bbox[1][1]]
            tmp_gt = []
            for gt_item in src_gt:
                if isin_bbox(gt_item, img_bbox):
                    bbox_item = [gt_item[0]-img_bbox[0][0], 
                                 gt_item[1]-img_bbox[0][1],
                                 gt_item[2]-img_bbox[0][0], 
                                 gt_item[3]-img_bbox[0][1]]
                    tmp_gt.append(bbox_item)
            out_img_list.append(tmp_img)
            out_gt_list.append(tmp_gt)
    return out_img_list, out_gt_list

def np_imread(img_file):
    cv_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def show_bbox(src_img,char_list):
    for tmp_gt in char_list:
        cv2.rectangle(src_img, (int(tmp_gt[1]), int(tmp_gt[0])),
                                    (int(tmp_gt[3]), int(tmp_gt[2])), (0,0,255), 3)
    src_img = cv2.resize(src_img, (512,512))
    cv2.imshow('tmp', src_img)
    cv2.waitKey()

def main(method):
    font_files = glob.glob('..\\..\\fonts\\*.ttf') + glob.glob('..\\..\\fonts\\*.TTF')
    
    font_file = random.choice(font_files)
    if method == 'stele':
        bg_files = glob.glob('..\\char_binary\\img_bgs\\碑帖*.png')
    elif method == 'calligraphy':
        bg_files = glob.glob('..\\char_binary\\img_bgs\\书法*.png')
    bg_file = random.choice(bg_files)
    bg_img = np_imread(bg_file)
    sentence_img, char_list = draw_sentence('混顿饭嘎口红放入奴隶撒娇更加开放而阿奴给你看路人粉结果撒额的按分而看就啊林父爱努蒂萨妇女节的萨夫家的沙发那u内容', bg_img, font_file, method)
    
    out_name = str(uuid.uuid1())
    out_img_file = os.path.join('', out_name+'.png')
    out_gt_file = os.path.join('', out_name+'.txt')
    cv2.imencode('.png', sentence_img)[1].tofile(out_img_file)
    # cv2.imwrite(out_img_file, sentence_img)
    img_list, gt_list = img_split(sentence_img, char_list)

    for img_index, img_item in enumerate(img_list):
        show_bbox(img_item, gt_list[img_index])

def add_noise(input_img, noise_level, noise_color):
    out_img = input_img
    return out_img

def draw_sentence(input_str, bg_img, font_file, method, noise_level=0):
    char_pos_list = []
    char_size = 100
    margin_size = 10
    bg_size = bg_img.shape[:2]
    bg_ratio = bg_size[0] / float(bg_size[1])
    line_num = int(math.sqrt(len(input_str) * bg_ratio)) + 1
    char_num_per_line = int(len(input_str) / line_num) + 1
    resize_ratio = max((line_num + 2) / bg_ratio, char_num_per_line + 2) * (char_size + margin_size) / float(min(bg_size))
    out_size = (int(bg_size[1] * resize_ratio) + 1, 
                int(bg_size[0] * resize_ratio) + 1)
    bg_img = cv2.resize(bg_img, out_size)
    if method == 'calligraphy':
        font_color = (random.randint(0,50), random.randint(0,50), random.randint(0,50))
    elif method == 'stele':
        font_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    for row_char in range(line_num):
        for col_char in range(char_num_per_line):
            index_char = row_char * char_num_per_line + col_char
            if index_char >= len(input_str):
                break
            char_val = input_str[index_char]
            start_pos = [(row_char + 1) * (char_size + margin_size),
                         (col_char + 1) * (char_size + margin_size)]
            char_bg = bg_img[start_pos[0]: start_pos[0]+128,
                             start_pos[1]: start_pos[1]+128]
            
            char_img, _, char_bbox = draw_chinese_char(char_val, char_bg, font_file, pos='random', font_color=font_color)
            # show_bbox(char_img, [char_bbox[0] + char_bbox[1]])

            bg_img[start_pos[0]: start_pos[0]+128,
                   start_pos[1]: start_pos[1]+128] = char_img
            char_pos = [start_pos[0] + char_bbox[0][0],
                        start_pos[1] + char_bbox[0][1],
                        start_pos[0] + char_bbox[1][0],
                        start_pos[1] + char_bbox[1][1]]
            char_pos_list.append(char_pos)
    return bg_img, char_pos_list

main(method='stele')
