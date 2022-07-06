import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

N_CLASSES = 4

def read_tfrecord(example):
    tfrecord_format = (
        {
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
    label = tf.cast(example["label"],tf.int32)
    label = tf.one_hot(label, N_CLASSES)
    return image, label

def load_dataset(filename:str, balanced:bool):
    ignore_order = tf.data.Options()
    if balanced:
        ignore_order.experimental_deterministic = True   # mantain original order
    else:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filename)  
    dataset = dataset.with_options(ignore_order)  
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def get_dataset(filename: str, batch_size: int, n_epochs:int, balanced:bool):
    dataset = load_dataset(filename, balanced)
    if not balanced: dataset = dataset.shuffle(128, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filename:str, batch_size:int, n_elems:int, n_epochs:int, balanced:bool):
        self.dataset = get_dataset(filename, batch_size, n_epochs, balanced)
        self.n = n_elems
        self.dataset = iter(self.dataset)
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n / self.batch_size))

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        'Generate one batch of data'
        imgs = []
        lbls = []
        images,labels = self.dataset.get_next()
        for i in range(len(images)):
            imgs.append(preprocess_input(images[i].numpy()))
            lbls.append(labels[i].numpy())
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        return imgs, lbls

def plot_hist(csv_path:str, save:bool, imgs_path:str=None)->list:
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        hist ={"epoch":[], "categorical_accuracy":[], "val_categorical_accuracy":[],"loss":[],"val_loss":[]}
        for row in csv_reader:
            hist["epoch"].append(float(row["epoch"]))
            hist["categorical_accuracy"].append(float(row["categorical_accuracy"]))
            hist["val_categorical_accuracy"].append(float(row["val_categorical_accuracy"]))
            hist["loss"].append(float(row["loss"]))
            hist["val_loss"].append(float(row["val_loss"]))

        n_epochs = len(hist["epoch"])

        plt.figure() 
        plt.title("LOSS")
        plt.plot(hist["loss"], color='blue', label='loss')
        plt.plot(hist["val_loss"], color='orange', label='Val_Loss')
        plt.xticks(range(0,n_epochs))
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="best")

        if save: 
            path_loss_fig = imgs_path+"/loss_{}.png".format(n_epochs)
            plt.savefig(path_loss_fig)
        else: plt.show()

        plt.figure()
        plt.title("ACCURACY")
        plt.plot(hist["categorical_accuracy"], color='blue', label='Accuracy')
        plt.plot(hist["val_categorical_accuracy"], color='orange',label='Val_accuracy')
        plt.xticks(range(0,n_epochs))
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="best")
        
        if save: 
            path_acc_fig = imgs_path+"/acc_{}.png".format(n_epochs)
            plt.savefig(path_acc_fig)
        else: plt.show()

        if save: return [path_loss_fig, path_acc_fig]