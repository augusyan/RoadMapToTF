# -*-coding:utf-8 -*-
# __author__=''
# function:
import os
import tensorflow as tf
from PIL import Image

# cwd = os.getcwd()
cwd = 'notMNIST_small/'


def create_record():
    writer = tf.python_io.TFRecordWriter("train_non_mninst.tfrecords")
    for index, name in enumerate(num_classes):
        class_path = cwd + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每一个图片的地址
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                # value=[index]决定了图片数据的类型label
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


if __name__ == '__main__':
    num_classes = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10}
    img, label = read_and_decode("train_non_mninst.tfrecords")

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    # 初始化所有的op
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l = sess.run([img_batch, label_batch])
            # l = to_categorical(l, 12)
            print(val.shape, l)
