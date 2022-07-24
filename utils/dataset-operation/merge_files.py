import os
import pathlib
from pathlib import Path
from dataset_dict import dataset_dict
import gdown

HOME_PATH = os.path.abspath('../gdown-files/')
SOURCEDATA_PATH = os.path.abspath(os.path.join(HOME_PATH,'Source'))

if not os.path.exists(SOURCEDATA_PATH):
    os.makedirs(SOURCEDATA_PATH)
    os.makedirs(os.path.join(SOURCEDATA_PATH, "dry"))
    os.makedirs(os.path.join(SOURCEDATA_PATH, "wet"))
    os.makedirs(os.path.join(SOURCEDATA_PATH, "snow"))

for index in range(len(dataset_dict)):

    print("### Zip Archive ", index+1, " of ", len(dataset_dict), " ###")
    print(dataset_dict[index])
    print("Zip Archive ID: ", dataset_dict[index]["id"])

    filename = dataset_dict[index]["name"]
    id = dataset_dict[index]["id"]

    name1, name2 = str(filename).split(".")
    print("Archive Name: ", name1)

    #print("Download Zip Archive...")
    #gdown.download(id=id, output=filename, quiet=False)
    print("Extract files...")
    os.system('unzip -q -o "%s" -d "%s"'%(filename, SOURCEDATA_PATH))

    print("Moving images...")
    print("Dry folder")
    os.system('find Source/home/%s/dry/ -name "*.jpg" -exec mv {} Source/dry \;'%name1)
    print("Wet folder")
    os.system('find Source/home/%s/wet/ -name "*.jpg" -exec mv {} Source/wet \;'%name1)
    print("Snow folder")
    os.system('find Source/home/%s/snow/ -name "*.jpg" -exec mv {} Source/snow \;'%name1)
    
    print("Done.\n")
    
 #Remember to add manually fog_snow.zip to the dataset.
