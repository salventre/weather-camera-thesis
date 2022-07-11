import os
import csv
import pathlib

HOME_PATH = os.path.abspath('../weather-camera-thesis/')
CSV_PATH = os.path.join(HOME_PATH,'data/dataset_annotation.csv')

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

dry_p = counter["dry"]/len_data
wet_p = counter["wet"]/len_data
snow_p = counter["snow"]/len_data
fog_p = counter["fog"]/len_data

#limiti = dry, wet, snow, fog --> 0.30, 0.29, 0.12, 0.29

print("Tot Images: ", len_data)
print(counter)
print("Verifica somma: ", counter["dry"] + counter["wet"] + counter["snow"] + counter["fog"], "\n")
print("Calcolo percentuale-> DRY: ", dry_p)
print("Calcolo percentuale-> WET: ", wet_p)
print("Calcolo percentuale-> SNOW: ", snow_p)
print("Calcolo percentuale-> FOG: ", fog_p)

print("Verifica percentuale: ", dry_p+wet_p+snow_p+fog_p)
