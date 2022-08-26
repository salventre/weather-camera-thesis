import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import numpy as np
import keras
sys.path.insert(0, os.path.abspath("/home/salvatore/Documenti/Weather Camera/weather-camera-thesis/src"))
from model import build_model
from classmap import *

CLASSIFICATOR_INPUT_SIZE = (224,224)

if __name__ == "__main__":

    print("Loading model...")
    model = build_model()
    model.load_weights("/home/salvatore/Documenti/Weather Camera/weather-camera-thesis/data/checkpoint/test_model.hdf5")
    print("Loading model...DONE")

    y_true = []
    y_pred = []
    
    print("Inference data:")

    img_path = "/home/salvatore/Documenti/Weather Camera/weather-camera-thesis/utils/Schermata del 2022-08-26 18-15-02.png"

    img = keras.preprocessing.image.load_img(img_path, target_size=CLASSIFICATOR_INPUT_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis


    true_label = category_index_classifier["fog"] #label of image

    predictions = model.predict(img_array)
    score = predictions[0]
    pred_label = np.argmax(score)
    print("True Label: ", category_map_classifier[true_label])
    print("Pred Label: ", category_map_classifier[pred_label])


