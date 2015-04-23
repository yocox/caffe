#!/usr/bin/python

import colorsys
import csv
import random
import os
import sys
import shutil
from PIL import Image

W = 16
H = 16

def gen_one_pic():
    h = random.random()
    im = Image.new("RGB", (W, H));
    for x in range(W):
        for y in range(H):
            h_ = h + random.random() * 0.4 - 0.2
            if h_ < 0:
                h_ += 1
            if h_ >= 1:
                h_ -= 1
            s = random.random()
            v = random.random()
            rgb = colorsys.hsv_to_rgb(h_, s, v)
            rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            im.putpixel((x, y), rgb)

    return (im, h)

def gen_pics(n):
    if os.path.exists('./hue_images'):
        shutil.rmtree('./hue_images')
    os.mkdir('./hue_images')
    fout_train = open('train_file_label.txt', 'wb')
    fout_test = open('test_file_label.txt', 'wb')
    bound = [i / 6.0 for i in range(6)]
    for i in range(n):
        (im, h) = gen_one_pic()
        tags = []
        r = abs(h - 0.00) < 0.22
        r = int(r or abs(h - 1.00) < 0.22)
        g = int(abs(h - 0.33) < 0.22)
        b = int(abs(h - 0.66) < 0.22)
        if r : tags.append('1')
        else : tags.append('-1')
        if g : tags.append('1')
        else : tags.append('-1')
        if b : tags.append('1')
        else : tags.append('-1')

        img_filename = './hue_images/%d.png' % i
        im.save(img_filename)
        tag_str = ' '.join(tags)
        if i < n * 24 / 25:
            fout_train.write('%s %s\n' % (img_filename, tag_str))
        else:
            fout_test.write('%s %s\n' % (img_filename, tag_str))

def convert_to_db():
    if os.path.exists('./multi-label-train.leveldb'):
        shutil.rmtree('./multi-label-train.leveldb')
    if os.path.exists('./multi-label-test.leveldb'):
        shutil.rmtree('./multi-label-test.leveldb')
    os.system('../../build/tools/convert_multi_tag_imageset.bin ./ train_file_label.txt multi-label-train.leveldb --backend leveldb')
    os.system('../../build/tools/convert_multi_tag_imageset.bin ./ test_file_label.txt multi-label-test.leveldb --backend leveldb')

def compute_mean():
    os.system('../../build/tools/compute_image_mean.bin multi-label-train.leveldb mean.binaryproto -backend leveldb')

if __name__ == '__main__':
    gen_pics(1000)
    convert_to_db()
    compute_mean()
