import os
import cv2
import subprocess
import numpy as np

HOME_PATH = os.path.abspath('../gdown-files/')
DATASET_PATH = os.path.abspath(os.path.join(HOME_PATH,'Dataset'))

CLASSIFICATOR_INPUT_SIZE = (224,224)

os.chdir(DATASET_PATH)
dirs = os.listdir(".")
print(DATASET_PATH)
print(dirs)

def resize(img:np.ndarray, target_shape:tuple)->np.ndarray:
    return cv2.resize(img, target_shape, interpolation = cv2.INTER_AREA)

def checkFile(imageFile):
    try:
        subprocess.run(["identify", "-regard-warnings", imageFile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).check_returncode()
        return True
    except (subprocess.CalledProcessError) as e:
        return False

def checkCorrupt_ResizeImages():
    for d in dirs:
        if d == "fog": #fog e snow già controllate e ridimensionate
            print("Folder: ", d)
            if os.path.isdir(d):
                id_dir = os.path.join(os.getcwd(), d)
                for f in os.listdir(id_dir):
                    image_path = os.path.join(id_dir, f)
                    img = cv2.imread(image_path)
                    if img is None:
                        os.system('rm "%s"'%image_path)
                    elif not checkFile(image_path):
                        os.system('rm "%s"'%image_path)
                    else:
                        img = resize(img, CLASSIFICATOR_INPUT_SIZE)
                        #cv2.imshow("", img)
                        #cv2.waitKey(0)
                        cv2.imwrite(image_path, img)



def show_image(image_path):           
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    cv2.imshow("", img)
    cv2.waitKey(0)

path = "/home/salvatore/Documenti/Weather Camera/gdown-files/wet/1543796772_122-0.jpg"

#show_image(path)
#checkCorrupt_ResizeImages()