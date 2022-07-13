import os
import pathlib
import csv

CODE_PATH = str(pathlib.Path(__file__).parent.absolute())
HOME_PATH = os.path.abspath('../weather-camera-thesis/')
CSV_PATH = os.path.join(os.path.abspath(os.path.join(pathlib.Path.cwd(), 'data')), "dataset_annotation.csv")
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Dataset'))

#HOME_PATH = os.path.abspath('../weather-camera-thesis/')
#pathlib.Path.cwd() current folder
#pathlib.Path.cwd().parent current parent folder

if not os.path.exists("data"):
    os.makedirs("data")

os.chdir(DATASET_PATH)
dirs = os.listdir(".")

with open(CSV_PATH, 'w', newline='') as file:
    fieldnames = ['imgPath', 'roadCondition']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    #writer.writeheader()

    for d in dirs:
        if os.path.isdir(d):
            id_dir = os.path.join(os.getcwd(), d)
            for f in os.listdir(id_dir):
                image_path = os.path.join(id_dir, f)
                writer.writerow({'imgPath': image_path, 'roadCondition': d})

print("Annotations generated !")