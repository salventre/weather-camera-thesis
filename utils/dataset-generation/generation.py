from fileinput import filename
import os
import pathlib
from pathlib import Path
from time import sleep
from copy_images import copy_images
from dataset_dict import dataset_dict

HOME_PATH = os.path.abspath('../dataset-generation/')
SOURCEDATA_PATH = os.path.abspath(os.path.join(HOME_PATH,'Source'))


for index in range(len(dataset_dict)):
    if not os.path.exists(SOURCEDATA_PATH):
        os.makedirs(SOURCEDATA_PATH)
    print("### Tar Archive ", index+1, " of ", len(dataset_dict), " ###")
    print(dataset_dict[index])
    print("Tar Archive URL: ", dataset_dict[index]["contentUrl"])

    filename = dataset_dict[index]["name"]
    url = dataset_dict[index]["contentUrl"]


    name1, name2 = str(filename).split(".")
    print("Archive Name: ", name1)

    print("Download Tar Archive...")
    os.system('wget -O "%s" %s -q --show-progress'%(os.path.join(SOURCEDATA_PATH,filename),url))
    print("Extract files...")
    os.system('tar -xf "%s" -C "%s"'%(os.path.join(SOURCEDATA_PATH,filename), SOURCEDATA_PATH))

    for root, dirs, files in os.walk(SOURCEDATA_PATH, topdown=True):
        for name in dirs:
            #print(os.path.join(root, name))
            if name == name1:
                #print("Found !")
                path_source = os.path.abspath(os.path.join(root, name))
            break
    print("Folder extraction Path found: ", path_source)

    print("Copying images...")
    copy_images(path_source, name1)
    print("Copying images... Finished !")
    os.system('rm -r "%s"'%SOURCEDATA_PATH)
    print("Cleaning Op. done. End\n")
