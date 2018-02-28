# -*- coding: utf-8 -*-
# @Time    : 18-2-26 上午11:17
# @Author  : Yan
# @Site    : 
# @File    : make_dataset.py
# @Software: PyCharm Community Edition
# @Function: Make tfrecords to tf
# @update: V0.1

"""制作tfrecords的工具类,图片放在文件夹里，label是文件名.其中的filename，即刚刚通过TFReader来生成的训练集。
通过将其转化成string类型数据，再通过reader来读取队列中的文件，并通过features的名字，‘label’和‘img_raw’来得
到对应的标签和图片数据。之后就是一系列的转码和reshape的工作了.
这里最最需要掌握的是
tf.python_io.TFRecordWriter(file_record)
tf.train.string_input_producer([filename])
tf.train.Example
writer.write(example.SerializeToString())
reader.read(filename_queue)
                    """

import os
import tensorflow as tf
from PIL import Image

# global define
# 数据集位置
path = "/17flowers/jpg"
output_pic = (224, 224)
record_file = "17flowers_train.tfrecords"


def load_data(path, resize_pics=(224, 224), file_record="train.tfrecords"):
    """
    :param path: Where U put data in the dir
    :param resize_pics: Pic size U need
    :param file_record: Output your tfrecords
    :return: None
    """
    cwd = os.getcwd()  # 获取当前的文件夹位置
    # print(cwd)
    writer = tf.python_io.TFRecordWriter(file_record)  # 写入train.tfrecords中
    NUM_FLIES = 0  # 计数器
    classes = os.listdir(cwd + path)  # label的文件列表
    classes = sorted(classes)
    # print(classes)
    for index, name in enumerate(classes):
        class_path = cwd + path + '/' + name + "/"
        print(class_path)
        if os.path.isdir(class_path):
            for img_name in sorted(os.listdir(class_path)):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize(resize_pics)
                img_raw = img.tobytes()  # 将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
                NUM_FLIES = NUM_FLIES + 1
                print(img_name, NUM_FLIES)
    writer.close()


def read_and_decode(filename):
    """
    根据文件名生成一个队列 
    :param filename: Where U put data in the dir
    :return: img, label
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 返回文件名和文件 
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string), })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)
    return img, label


def test_record(data_name):
    # 测试tfrecord是否有效
    for serialized_example in tf.python_io.tf_record_iterator(data_name):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['image'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
        print(image, label)


# 调用load_data()生成tfrecord
# load_data(path,output_pic,file_record=record_file)

# 解码输出img,label
img, label = read_and_decode(record_file)
# 使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    for i in range(3):
        val, l= sess.run([img_batch, label_batch])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12)
        print(val.shape, l)
    coord.request_stop()
    coord.join()

