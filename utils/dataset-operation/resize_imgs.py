import imp
import os
import pathlib
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

HOME_PATH = os.path.abspath('../gdown-files/')
DATASET_PATH = os.path.abspath(os.path.join(HOME_PATH,'Dataset'))

CLASSIFICATOR_INPUT_SIZE = (224,224)

os.chdir(DATASET_PATH)
dirs = os.listdir(".")

def resize(img:np.ndarray, target_shape:tuple)->np.ndarray:
    return cv2.resize(img, target_shape, interpolation = cv2.INTER_AREA)

def resize_imgs_CV():
    for d in dirs:
        print("Folder: ", d)
        if os.path.isdir(d):
            id_dir = os.path.join(os.getcwd(), d)
            for f in os.listdir(id_dir):
                image_path = os.path.join(id_dir, f)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        os.system('rm "%s"'%image_path)
                        print("Img removed: ", image_path)


                    else:
                        #cv2.imshow("",img)
                        #cv2.waitKey(0)
                        img = resize(img, CLASSIFICATOR_INPUT_SIZE)
                        #cv2.imshow("", img)
                        #cv2.waitKey(0)
                        cv2.imwrite(image_path, img)
                except:
                    os.system('rm "%s"'%image_path)
                    print("Img removed: ", image_path)


def resize_imgs_PIL():
    for d in dirs:
        print("Folder: ", d)
        if os.path.isdir(d):
            id_dir = os.path.join(os.getcwd(), d)
            for f in tqdm(os.listdir(id_dir)):
                image_path = os.path.join(id_dir, f)
                try:
                    imgP = Image.open(image_path)
                    imgC = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    imgP.verify()
                    if imgC is None:
                        print("ImgC is none: ", image_path)

                    #img.show()
                    #img2 = img.resize((CLASSIFICATOR_INPUT_SIZE), Image.ANTIALIAS)
                    #img2.show()
                except Exception:
                    print("imgP raises an exception: ", image_path)