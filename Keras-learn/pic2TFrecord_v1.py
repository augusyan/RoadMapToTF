# -*-coding:utf-8 -*-
# __author__=''
# function:
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#路径
cwd='notMNIST_small/'
#类别
classes={'A':1, 'B':2 ,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10}
#tfrecords格式文件名
writer= tf.python_io.TFRecordWriter("nonMNIST.tfrecords")

for index,name in enumerate(classes):
    class_path=cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name #每一个图片的地址

        img=Image.open(img_path)
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            #value=[index]决定了图片数据的类型label
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串

writer.close()