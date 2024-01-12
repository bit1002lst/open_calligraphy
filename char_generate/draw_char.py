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
        lt_pos = ((128-word_width)/2, (128-word_height)/2)
        rb_pos = (lt_pos[0] + word_width, lt_pos[1] + word_height)
    if font_color is None:
        font_color = (random.randint(0,50), random.randint(0,50), random.randint(0,50))
    draw.text(lt_pos, input_char, fill = font_color, font=ft)
    draw_gt.text(lt_pos, input_char, fill = (0, 0, 0), font=ft)
    im_tmp = np.array(im_tmp)
    im_gt = np.array(im_gt)
    kernel_size = (random.randint(0,7)*2+1, random.randint(0,7)*2+1)
    cv2.GaussianBlur(im_tmp, kernel_size, 0, 1)
    return im_tmp, im_gt, [lt_pos, rb_pos]

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

def main():
    font_files = glob.glob('../../fonts/*.ttf') + glob.glob('../../fonts/*.TTF')
    font_file = random.choice(font_files)
    bg_files = glob.glob('../char_binary/img_bgs/兰亭集序_bg.png')
    bg_file = random.choice(bg_files)
    bg_img = cv2.imread(bg_file)
    # char_img, gt_img = draw_chinese_char('我', bg_file, font_file)
    # cv2.imshow('src', char_img)
    # cv2.imshow('gt', gt_img)
    # cv2.waitKey()
    print(bg_file)
    draw_sentence('混顿饭嘎口红放入奴隶撒娇更加开放而阿奴给你看路人粉结果撒额的按分而看就啊林父爱', bg_img, font_file)
    

def draw_sentence(input_str, bg_img, font_file):
    gt_list = []
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
    font_color = (random.randint(0,50), random.randint(0,50), random.randint(0,50))
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
            bg_img[start_pos[0]: start_pos[0]+128,
                   start_pos[1]: start_pos[1]+128] = char_img
            cv2.imshow('tmp', bg_img)
            cv2.waitKey()

main()

