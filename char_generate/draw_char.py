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

def draw_chinese_char(input_char, bg_img, font):
    # im = Image.new('1', (128, 128), 255)
    im = Image.open(bg_img)
    src_img = cv2.imread(bg_img)
    crop_size = random.randint(50, min(src_img.shape[:2]))
    crop_x = random.randint(0, src_img.shape[0]-crop_size)
    crop_y = random.randint(0, src_img.shape[1]-crop_size)
    im_tmp = src_img[crop_x:crop_x+crop_size,
                crop_y:crop_y+crop_size]
    im_tmp = cv2.resize(im_tmp, (128, 128))
    im_tmp = Image.fromarray(im_tmp)

    draw = ImageDraw.Draw(im_tmp) #修改图片
            
    ft = ImageFont.truetype(
                font=font,
                size=100
            )

    word_width, word_height = draw.textsize(input_char, font=ft)
    draw.text(((128-word_width)/2,(128-word_height)/2), input_char, fill = (random.randint(0,50), random.randint(0,50), random.randint(0,50)), font=ft)
    im_tmp = np.array(im_tmp)
    kernel_size = (random.randint(0,7)*2+1, random.randint(0,7)*2+1)
    cv2.GaussianBlur(im_tmp, kernel_size, 0, 1)
    cv2.imwrite('tmp.png', im_tmp)

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
    bg_files = glob.glob('../char_binary/img_bgs/*.png')
    bg_file = random.choice(bg_files)
    draw_chinese_char('我', bg_file, font_file)


main()

