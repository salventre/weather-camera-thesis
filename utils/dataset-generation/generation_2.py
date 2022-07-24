from fileinput import filename
import os
from pathlib import Path
from copy_images import copy_images
from dataset_dict import dataset_dict

HOME_PATH = os.path.abspath('../dataset-generation/')
SOURCEDATA_PATH = os.path.abspath(os.path.join(HOME_PATH,'Source'))

i = 71

for index in dataset_dict[71:201]:
    if not os.path.exists(SOURCEDATA_PATH):
        os.makedirs(SOURCEDATA_PATH)
    print("### Tar Archive ", i, " of ", len(dataset_dict), " ###")
    print(index)
    print("Tar Archive URL: ", index["contentUrl"])

    filename = index["name"]
    url = index["contentUrl"]


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
    i = i+1