import os
import csv
import pathlib

HOME_PATH = os.path.abspath('../weather-camera-thesis/')
CSV_PATH = os.path.join(os.path.abspath(os.path.join(pathlib.Path.cwd(), 'data')), "dataset_annotation.csv")

#HOME_PATH = os.path.abspath('../weather-camera-thesis/')
#pathlib.Path.cwd() current folder
#pathlib.Path.cwd().parent current parent folder

classes = ["dry", "wet", "snow", "fog"]
counter = {c:0 for c in classes}
len_data = 0

with open(CSV_PATH, 'r') as file:
    reader = csv.reader(file, delimiter = ',')
    for row in reader:
        c = row[1]
        counter[c] += 1
        len_data += 1

DRY_P = counter["dry"]/len_data
WET_P = counter["wet"]/len_data
SNOW_P = counter["snow"]/len_data
FOG_P= counter["fog"]/len_data

#limiti = dry, wet, snow, fog --> 0.30, 0.29, 0.12, 0.29

print("Total Images: ", len_data)
print(counter)
print("Check Sum: ", counter["dry"] + counter["wet"] + counter["snow"] + counter["fog"], "\n")
print("Percentage calculation:")
print("DRY: ", round(DRY_P, 4)*100, "%")
print("WET: ", round(WET_P, 4)*100, "%")
print("SNOW: ", round(SNOW_P, 4)*100, "%")
print("FOG: ", round(FOG_P, 4)*100, "%")

print("Check Percentage: ", (DRY_P + WET_P + SNOW_P + FOG_P)*100, "%")
