import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
from classmap import category_index_classifier

CODE_PATH = str(pathlib.Path(__file__).parent.absolute())
HOME_PATH = os.path.abspath('../weather-camera-thesis/')
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Dataset'))
TFRECORD_PATH = os.path.join(HOME_PATH,'data/tfrecord')

def class_text_to_int(row_label):
    return category_index_classifier[row_label]


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example(image, image_path, label):
    feature = {
        "image/filename": bytes_feature(image_path.encode('utf8')),
        "image/encoded": bytes_feature(tf.image.encode_jpeg(image)),
        "label": int64_feature(class_text_to_int(label)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecord(writer, image_path, label):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #print(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    example = create_example(image, image_path, label)
    writer.write(example.SerializeToString())

def generate_tfrecords():

    os.chdir(DATASET_PATH)
    dirs = os.listdir(".")

    tfrecord_file_name = "tfrecordexample.record"
    writer = tf.io.TFRecordWriter(os.path.join(TFRECORD_PATH, tfrecord_file_name))

    for d in dirs:
        if os.path.isdir(d):
            id_dir = os.path.join(os.getcwd(), d)
            for f in os.listdir(id_dir):
                image_path = os.path.join(id_dir, f)
                write_tfrecord(writer,image_path, d)

generate_tfrecords()