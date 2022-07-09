import os
import pathlib
import cv2
import numpy as np

CODE_PATH = str(pathlib.Path(__file__).parent.absolute())
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Dataset'))

CLASSIFICATOR_INPUT_SIZE = (224,224)

def resize(img:np.ndarray, target_shape:tuple)->np.ndarray:
    return cv2.resize(img, target_shape, interpolation = cv2.INTER_AREA)

def resize_imgs():

    os.chdir(DATASET_PATH)
    dirs = os.listdir(".")

    for d in dirs:
        if os.path.isdir(d):
            id_dir = os.path.join(os.getcwd(), d)
            for f in os.listdir(id_dir):
                image_path = os.path.join(id_dir, f)
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                img = resize(img, CLASSIFICATOR_INPUT_SIZE)
                cv2.imwrite(image_path, img)

resize_imgs()