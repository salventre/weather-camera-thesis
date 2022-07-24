import os
import pathlib
from pathlib import Path
import csv

CODE_PATH = str(pathlib.Path(__file__).parent.absolute())
CSV_PATH = os.path.abspath(os.path.join(CODE_PATH,'images.csv'))

def copy_images(path_source, station_source):
    with open(CSV_PATH, 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            stationID = dict(row)["stationId"]
            if stationID == station_source:
                imgPath = dict(row)["imgPath"]
                roadC = dict(row)["roadCondition"]
                #print(stationID, " & ",imgPath, " & ", roadC)
                path1, path2, path3 = str(imgPath).split('/')
                
                #print("\nPath1:",path1)
                #print("\nPath2:", path2)
                #print("\nPath3:",path3)

                if path2[0] == '0':
                    path2 = path2[1]
                
                IMAGESOURCE_PATH_FOLDER = os.path.join(path_source, path2)
                IMAGESOURCE_PATH_NAME = os.path.join(IMAGESOURCE_PATH_FOLDER, path3)
                
                if roadC == "dry":
                    NEW_PATH_FOLDER = os.path.join(CODE_PATH,'dry')
                    NEW_IMAGE_PATH = os.path.join(NEW_PATH_FOLDER, path3)
                    Path(IMAGESOURCE_PATH_NAME).rename(NEW_IMAGE_PATH)
                elif roadC == "wet":
                    NEW_PATH_FOLDER = os.path.join(CODE_PATH,'wet')
                    NEW_IMAGE_PATH = os.path.join(NEW_PATH_FOLDER, path3)
                    Path(IMAGESOURCE_PATH_NAME).rename(NEW_IMAGE_PATH)
                elif roadC == "snow":
                    NEW_PATH_FOLDER = os.path.join(CODE_PATH,'snow')
                    NEW_IMAGE_PATH = os.path.join(NEW_PATH_FOLDER, path3)
                    Path(IMAGESOURCE_PATH_NAME).rename(NEW_IMAGE_PATH)
        
