import pandas as pd
from tqdm import tqdm
import os
from model import build_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from classmap import category_index_classifier
import keras
import argparse

CLASSIFICATOR_INPUT_SIZE = (224,224)
TEST_SET_PATH = os.path.abspath('../weather-camera-thesis/src/partition_info/test_filenames.txt')
OUTPUT_PATH = os.path.abspath('../weather-camera-thesis/data/doc/results.txt')

def init_parameter():   
    parser = argparse.ArgumentParser(description='File for Testing Classifier')
    parser.add_argument("--dataset", type=str, default=TEST_SET_PATH, help="Path della cartella di filenames-test")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path file dei risultati")
    parser.add_argument("--model_path", type=str, help="Path del modello del classificatore")      
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = init_parameter()

    print("Loading model...")
    model = build_model()
    model.load_weights(args.model_path)
    print("Loading model...DONE")

    y_true = []
    y_pred = []
    
    with open(args.dataset) as f:
        lines = f.readlines()
        lines = [f.rstrip() for f in lines]

    print("Inference data:")
    for f in tqdm(lines): #verify this
        path1, path2 = f.split(',')
        filename = [path1[2:-1],path2[2:-2]]

        img_path = filename[0]

        img = keras.preprocessing.image.load_img(img_path, target_size=CLASSIFICATOR_INPUT_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        
        true_label = category_index_classifier[filename[1]]
        y_true.append(true_label)

        predictions = model.predict(img_array)
        score = predictions[0]
        pred_label = np.argmax(score)
        y_pred.append(pred_label)


    labels = list(category_index_classifier.keys())

    print("Building confusion matrix...")
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.abspath('../weather-camera-thesis/data/doc/cm_validation.png'))
    matrix = pd.DataFrame(matrix, index=["true:{}".format(x) for x in labels], columns=["pred:{}".format(x) for x in labels])
    print("Building confusion matrix...DONE")
    print("Confusion Matrix:\n", matrix)
    
    print("Computing Metrics...")
    acc = accuracy_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred, average='weighted')
    prec_score = precision_score(y_true, y_pred, average='weighted')
    f1score = f1_score(y_true, y_pred, average='weighted')
    print("Computing Metrics...DONE")
    print("Accuracy Score: ", round(acc, 4))
    print("Recall Score: ", round(rec_score, 4))
    print("Precision Score: ", round(prec_score, 4))
    print("F1-Score: ", round(f1score, 4))

    with open(args.output_path,"w") as f:
        f.write("CONFUSION MATRIX\n"+matrix.to_string())
        f.write("\n\nACCURACY:\n"+str(round(acc, 4)))
        f.write("\n\nPRECISION:\n"+str(round(prec_score, 4)))
        f.write("\n\nRECALL:\n"+str(round(rec_score, 4)))
        f.write("\n\nF1-SCORE:\n"+str(round(f1score, 4)))

