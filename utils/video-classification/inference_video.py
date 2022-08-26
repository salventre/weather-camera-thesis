from collections import deque
import os
import cv2
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import keras
import numpy as np
sys.path.insert(0, os.path.abspath("/home/salvatore/Documenti/Weather Camera/weather-camera-thesis/src"))
from model import build_model
from classmap import category_map_classifier

IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASSIFICATOR_INPUT_SIZE = (224,224)
CLASSES_LIST = [0, 1, 2, 3]
NUM_CLASSES = 4

MODEL_FOLDER = os.path.abspath('./data/checkpoint/')
model_name = "test_model.hdf5" #modify to choose the model

MODEL_PATH = os.path.abspath(os.path.join(MODEL_FOLDER, model_name))

def predict_on_live_video(model, video_file_path, output_file_path, window_size):

    predicted_labels_probabilities_deque = deque(maxlen = window_size)
    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (original_video_width, original_video_height))

    while True: 

        ret, frame = video_reader.read() 

        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))

        img_array = keras.preprocessing.image.img_to_array(resized_frame)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
  
        predicted_labels_probabilities = model.predict(img_array)
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        if len(predicted_labels_probabilities_deque) == window_size:
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            predicted_class_name = category_map_classifier[predicted_label]
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        video_writer.write(frame)
 
    video_reader.release()
    video_writer.release()

def make_average_predictions(model, video_file_path, predictions_frames_count):
    
    predicted_labels_probabilities_np = np.zeros((predictions_frames_count, NUM_CLASSES), dtype = float)

    video_reader = cv2.VideoCapture(video_file_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = video_frames_count // predictions_frames_count

    for frame_counter in range(predictions_frames_count): 

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        _ , frame = video_reader.read() 
        resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))   
        img_array = keras.preprocessing.image.img_to_array(resized_frame)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
  
        predicted_labels_probabilities = model.predict(img_array)
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities

    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]

    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:
        predicted_class_name = category_map_classifier[predicted_label]
        predicted_probability = predicted_labels_probabilities_averaged[predicted_label]
        print(f"Class Name: {predicted_class_name}   Avg. Probability: {(predicted_probability*100):.5} %")

    video_reader.release()


if __name__ == "__main__":

    print("Loading model...")
    model = build_model()
    model.load_weights(MODEL_PATH)
    print("Loading model...DONE")

    output_directory = os.path.abspath('./utils/video-classification/output_video')
    os.makedirs(output_directory, exist_ok = True)

    input_video_folder = os.path.abspath("./utils/video-classification/input_video")
    video_title = 'Dry3.mp4'

    input_video_file_path = os.path.abspath(os.path.join(input_video_folder, video_title))
 
    frame_window_size = 100
    output_video_file_path = f'{output_directory}/{video_title}-Output-WSize-{frame_window_size}.mp4'
    print("\nPredict Live Video...")
    predict_on_live_video(model, input_video_file_path, output_video_file_path, frame_window_size)
    print("Predict Live Video...DONE - Video ready !\n")

    # To get a single prediction for the entire video
    predictions_frames_count = 100
    print("Single Prediction for entire video:")
    make_average_predictions(model, input_video_file_path, predictions_frames_count)