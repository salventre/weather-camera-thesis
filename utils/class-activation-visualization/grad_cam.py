import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
import sys
import tensorflow as tf
from tensorflow import keras
sys.path.insert(0, os.path.abspath("/home/salvatore/Documenti/Weather Camera/weather-camera-thesis/src"))
from model import build_model
from classmap import *

# Display
import cv2
import matplotlib.cm as cm


"""
## Configurable parameters

You can change these to another model.

To get the values for `last_conv_layer_name` use `model.summary()`
to see the names of all layers in the model.

"""

MODEL_FOLDER = os.path.abspath('./data/checkpoint/')
model_name = "model-epoch_06.hdf5" #modify to choose the model

MODEL_PATH = os.path.abspath(os.path.join(MODEL_FOLDER, model_name))


img_size = (224, 224)
last_conv_layer_name = "Conv_1_bn"


def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)

    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM 
    image = cv2.imread(cam_path)
    cv2.imshow('grad_cam', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    print("Loading model...")
    model = build_model()
    model.load_weights(MODEL_PATH)
    print("Loading model...DONE")

    output_directory = os.path.abspath('./utils/class-activation-visualization/output_image')
    os.makedirs(output_directory, exist_ok = True)
    

    input_image_folder = os.path.abspath("./utils/class-activation-visualization/input_image")
    image_title = 'fogImage.png'

    input_image_file_path = os.path.abspath(os.path.join(input_image_folder, image_title))
 

    # Prepare image
    img_array = get_img_array(input_image_file_path, size=img_size)

    preds = model.predict(img_array)
    score = preds[0]
    pred_label = np.argmax(score)
    print("Predicted:", category_map_classifier[pred_label])

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=category_index_classifier["fog"])
    output_image_file_path = f'{output_directory}/{image_title}-Grad_Cam.jpg'
    save_and_display_gradcam(input_image_file_path, heatmap, output_image_file_path)
