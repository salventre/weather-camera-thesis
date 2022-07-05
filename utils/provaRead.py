import os
import csv
import pathlib

CODE_PATH = str(pathlib.Path(__file__).parent.absolute())
CSV_PATH = os.path.join(CODE_PATH,'CSVfromImages.csv')

#HOME_PATH = os.path.abspath('../weather-camera-thesis/')
#pathlib.Path.cwd() current folder
#pathlib.Path.cwd().parent current parent folder

with open(CSV_PATH, 'r') as file:
    csv_file = csv.DictReader(file)
    for row in csv_file:
        imgPath = dict(row)["imgPath"]
        roadC = dict(row)["roadCondition"]
        
        print("\nPath:", imgPath)
        print("\nRoad Condition (label): ", roadC)
        break