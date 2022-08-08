import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
from model import build_model
import tensorflow as tf
import argparse
from plot_hist import *
from sklearn.utils import class_weight
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

NEW_DATASET_PATH = os.path.abspath('../weather-camera-thesis/data/dataset/train/')

def init_parameter()->argparse.Namespace:   
    parser = argparse.ArgumentParser(description='Classifier Training Script')
    parser.add_argument("--train_path", type=str, default=NEW_DATASET_PATH, help="Path della cartella contenente i file di training")
    parser.add_argument("--learning_rate", type=float, default=0.00003, help="Learning_rate iniziale")
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione dei batches")
    args = parser.parse_args()
    return args

def train(args:argparse.Namespace):
    if not os.path.exists('./data/checkpoint'): os.makedirs('./data/checkpoint')
    if not os.path.exists('./data/logs'): os.makedirs('./data/logs')
    if not os.path.exists('./data/tb_logs'): os.makedirs('./data/tb_logs')
    if not os.path.exists('./data/doc'): os.makedirs('./data/doc')
    
    TRAIN_DIM = 7917 #1454724
    VAL_DIM = 1977  #363680
    CLASSIFICATOR_INPUT_SIZE = (224,224)

    batch_size = args.batch_size
    epochs = args.epochs
    patience= 8
    learning_rate = args.learning_rate

    freeze_model = False
    resume_training = False
    model_path = os.path.abspath("/") #to check
    print("--- Training Parameters: ---")
    print("TRAIN Dim: ", TRAIN_DIM, ", VAL Dim: ", VAL_DIM)
    print("Batch Size: ", batch_size, " - Epochs", epochs, " - Learning Rate: ", learning_rate)
    print("Patience: ", patience, " - Freeze Model: ", freeze_model, " - Resume Training: ", resume_training,"\n")

    # model
    model = build_model(freeze_model)
    if resume_training:
        model.load_weights(model_path)
    
    #lr_schedule = tf.keras.optimizers.schedules.CosineDecay(args.learning_rate, int(args.epochs*(TRAIN_DIM/args.batch_size)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                loss=tf.keras.losses.categorical_crossentropy, 
                metrics=[tf.keras.metrics.categorical_accuracy]
                )

    train_datagen = ImageDataGenerator(validation_split=0.20)
    #test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory=NEW_DATASET_PATH,
        target_size=CLASSIFICATOR_INPUT_SIZE,
        #color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=42
    )

    valid_generator = train_datagen.flow_from_directory(
        directory=NEW_DATASET_PATH,
        target_size=CLASSIFICATOR_INPUT_SIZE,
        #color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='validation',
        shuffle=True,
        seed=42
    )

    '''
    test_generator = test_datagen.flow_from_directory(
        directory="/home/salvatore/Documenti/Weather Camera/weather-camera-thesis/data/dataset/test",
        target_size=CLASSIFICATOR_INPUT_SIZE,
        #color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )'''

    class_weights = dict(zip(np.unique(valid_generator.classes), class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(valid_generator.classes), y=valid_generator.classes))) 
    print("Class Weights: ", class_weights)
    
    #resize images
    #train_ds = train_ds.map(lambda image, label: (tf.image.resize(image, CLASSIFICATOR_INPUT_SIZE), label))
    #val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, CLASSIFICATOR_INPUT_SIZE), label))
    #train_ds = train_ds.prefetch(buffer_size=32)
    #val_ds = val_ds.prefetch(buffer_size=32)

    # callbacks --> val_loss, prima c'era "val_categorical_accuracy"
    filepath = 'model-epoch_{epoch:02d}.hdf5'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join('./data/checkpoint', filepath), save_weights_only=True, verbose=1, save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience,verbose=1,monitor='val_loss',mode='auto')
    history_logger_cb = tf.keras.callbacks.CSVLogger('./data/logs/training_log.csv', separator=",", append=True)
    tensorboad_cb = tf.keras.callbacks.TensorBoard(log_dir="./data/tb_logs", write_graph=False)
    callbacks = [checkpoint_cb,early_stopping_cb,history_logger_cb,tensorboad_cb]

    hist = model.fit(train_generator, validation_data = valid_generator, epochs=epochs, verbose = 1, callbacks=callbacks, shuffle = False, class_weight=class_weights,
                steps_per_epoch=int(np.ceil(TRAIN_DIM / batch_size)),validation_steps=int(np.ceil(VAL_DIM / batch_size)))


if __name__ == "__main__":

    args = init_parameter()
    train(args)
