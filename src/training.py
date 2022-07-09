import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
from model import build_model
import tensorflow as tf
import argparse
from utils import *

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Dataset'))


def init_parameter()->argparse.Namespace:   
    parser = argparse.ArgumentParser(description='Classifier training script')
    parser.add_argument("--train_path", type=str, default='tfrecords', help="Path della cartella contenente i file tfrecords di training")
    parser.add_argument("--valid_path", type=str, default='tfrecords', help="Path della cartella contenente i file tfrecords di validation")
    parser.add_argument("--checkpoint_path", type=str, default='./data/checkpoint', help="Path della cartella in cui salvare i checkpoint del modello")
    parser.add_argument("--logging_path", type=str, default='./data/logs', help="Path della cartella in cui salvare i logs del training")
    parser.add_argument("--tb_path", type=str, default='./data/tb_logs', help="Path della cartella in cui salvare i logs di Tensorboard")
    parser.add_argument("--learning_rate", type=float, default=0.00003, help="Learning_rate iniziale")
    parser.add_argument("--epochs", type=int, default=70, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione dei batches")
    parser.add_argument("--resume_training", action="store_true", default=False, help="riprendere un training iniziato precedentemente.")
    parser.add_argument("--model_path", type=str, default=None, help="Nome del modello da caricare nel caso in cui resume_training sia posto a True")
    parser.add_argument("--balanced", action="store_true", help="Se usare batches bilanciate")
    args = parser.parse_args()
    return args

def train(args:argparse.Namespace):
    if not os.path.exists(args.checkpoint_path): os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.logging_path): os.makedirs(args.logging_path)
    if not os.path.exists(args.tb_path): os.makedirs(args.tb_path)
    
    # model
    model = build_model()
    if args.resume_training:
        model.load_weights(args.model_path)
    
    #lr_schedule = tf.keras.optimizers.schedules.CosineDecay(args.learning_rate, int(args.epochs*(TRAIN_DIM/args.batch_size)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), 
                loss=tf.keras.losses.categorical_crossentropy, 
                metrics=[tf.keras.metrics.categorical_accuracy]
                )
    
    # generators
    #filenames = glob.glob(args.train_path+"/*.tfrecords")
    #train_gen = DataGenerator(filenames, args.batch_size, TRAIN_DIM, args.epochs, args.balanced)
    #filenames = glob.glob(args.valid_path+"/*.tfrecords")
    #valid_gen = DataGenerator(filenames, args.batch_size, VAL_DIM, args.epochs, args.balanced)

    CLASSIFICATOR_INPUT_SIZE = (224,224)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        label_mode= 'categorical',
        seed=1337,
        image_size=CLASSIFICATOR_INPUT_SIZE,
        batch_size=batch_size,)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        label_mode= 'categorical',
        seed=1337,
        image_size=CLASSIFICATOR_INPUT_SIZE,
        batch_size=batch_size,
    )

    #resize images
    train_ds = train_ds.map(lambda image, label: (tf.image.resize(image, CLASSIFICATOR_INPUT_SIZE), label))
    val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, CLASSIFICATOR_INPUT_SIZE), label))

    
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    # callbacks
    filepath = 'model-epoch_{epoch:02d}.hdf5'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.checkpoint_path, filepath), save_weights_only=True, verbose=1, save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,verbose=1,monitor='val_categorical_accuracy',mode='max')
    history_logger_cb = tf.keras.callbacks.CSVLogger('./data/logs/training_log.csv', separator=",", append=True)
    tensorboad_cb = tf.keras.callbacks.TensorBoard(log_dir="./data/tb_logs", write_graph=False)
    callbacks = [checkpoint_cb,early_stopping_cb,history_logger_cb,tensorboad_cb]

    hist = model.fit(train_ds, validation_data = val_ds, epochs=args.epochs, verbose = 1, callbacks=callbacks, shuffle = False) 
                        #),steps_per_epoch=int(np.ceil(TRAIN_DIM / args.batch_size)),validation_steps=int(np.ceil(VAL_DIM / args.batch_size)))

    plot_hist(hist)

if __name__ == "__main__":
    #tf.enable_eager_execution()
    args = init_parameter()
    #base_path = "../../datasets_processing/classifier_training/stats/"
    #TRAIN_DIM = 31770
    #VAL_DIM = 7942
    #print("TRAIN: ", TRAIN_DIM, " , VAL: ", VAL_DIM)
    train(args)