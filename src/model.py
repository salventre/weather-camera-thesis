import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow import keras


INPUT_SHAPE = (224,224,3)
IMG_SIZE = 224
N_CLASSES = 4 #dry, wet, snow, fog
N_LAYERS_TO_TRAIN = 10 #to modify

img_augmentation = keras.Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model(freeze:bool=None)->tf.keras.Model:

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    base_model = MobileNetV2(INPUT_SHAPE, include_top=False, input_tensor=x, weights='imagenet') # 154 layers
    if freeze is not None:
        if freeze: base_model.trainable = False
        else:
            for layer in base_model.layers:
                if not isinstance(layer, layers.BatchNormalization):
                    layer.trainable = True
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)
    model = layers.Dense(N_CLASSES, 'softmax')(x)
    final_model = tf.keras.Model(inputs=base_model.input, outputs=model)
    #print(final_model.summary())
    return final_model