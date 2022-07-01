import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2 as cv
import numpy as np
from classmap import category_map_classifier as classmap
import glob
import json
import pandas as pd


CODE_PATH = str(pathlib.Path(__file__).parent.absolute())
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Dataset'))
TFRECORD_PATH = os.path.join(CODE_PATH,'tfrecord')

tfrecord_file_name = "tfrecordexample.record"
TFRECORD_FILE_PATH = os.path.join(TFRECORD_PATH, tfrecord_file_name)

def count_tfrecord(path_tfrecord, out):
    filenames = glob.glob(path_tfrecord+"/*.tfrecords")
    print(len(filenames), " files founded")
    counter = {f:0 for f in filenames}

    for f in filenames:
      # count number of occurences in the tfrecord
      dataset = tf.data.TFRecordDataset(f)
      cnt = dataset.reduce(np.int64(0), lambda x, _: x+1)
      counter[f] = int(cnt.numpy())
      print("Done ", f)
    
    with open(out,"w") as f:
      json.dump(counter,f)

def _get_format():
    tfrecord_format = (
        {
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
    )

    return tfrecord_format

def _extract_fn(tfrecord):
    # Extract features using the keys set during creation
    tfrecord_format = _get_format()
    
    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, tfrecord_format)
    image = tf.image.decode_jpeg(sample["image/encoded"], channels=3)
    label = sample['label']

    return image, label

def read_tfrecord(show):
    if not show:
        stats = { 
            "samples": 0, 
            "dry":0, 
            "wet":0, 
            "fog":0, 
            "snow":0,
            "other":0
        }
    #filenames = glob.glob(path_tfrecord+"/*.tfrecords")
    #print(len(filenames), " files founded")
    dataset = tf.data.TFRecordDataset(TFRECORD_FILE_PATH)
    dataset = dataset.map(_extract_fn,num_parallel_calls=1)
    dataset = dataset.shuffle(128, seed=123, reshuffle_each_iteration=False)
    iterator = iter(dataset)

    # count number of occurences in the tfrecord
    # cnt = dataset.reduce(np.int64(0), lambda x, _: x+1)
    # print("{} records".format(cnt.numpy()))

    done = False
    while not done:
        try:
            image, label = iterator.get_next()
            label = label.numpy()
            label_str = classmap[int(label)]
            image = image.numpy()
            if show:
                print(label_str)
                cv.imshow("figure",image)
                cv.waitKey(0)
            else:
                stats["samples"]+=1
                stats[label_str]+=1
        except tf.errors.OutOfRangeError:
            done=True
    
    df = pd.DataFrame(stats, index=[0])
    df.to_csv(CODE_PATH + "./stats.csv", index=False)


read_tfrecord(False)
