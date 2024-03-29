import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from tensorflow import keras

N_CLASSES = 4
image_size = (224, 224)

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


#dict_keys(['loss', 'categorical_accuracy', 'val_loss', 'val_categorical_accuracy'])

def plot_hist_live(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    plt.plot(hist.history["categorical_accuracy"])
    plt.plot(hist.history["val_categorical_accuracy"])
    plt.title("Model C_Acc")
    plt.ylabel("C_acc")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()