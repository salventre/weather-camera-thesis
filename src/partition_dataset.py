from copy import deepcopy
import os
import shutil
import math
import random
from tqdm import tqdm
import csv
import pathlib
#from read_csv_asnnotation import DRY_P, WET_P, SNOW_P, FOG_P


HOME_PATH = os.path.abspath('../weather-camera-thesis/')
CSV_PATH = os.path.join(os.path.abspath(os.path.join(pathlib.Path.cwd(), 'data')), "dataset_annotation.csv")
CODE_PATH = str(pathlib.Path(__file__).parent.absolute())
NEW_DATASET_PATH = os.path.abspath('../weather-camera-thesis/data/dataset/')


def print_filenames(path:str, filenames:list)->None:
    with open(path,"w") as f:
        for item in filenames:
            f.write("%s\n" % item)


def check_ds_distribution(counter:dict)->bool:
    # check if the new created dataset respects the orginal dataset distribution #
    #limits = dry, wet, snow, fog --> 0.30, 0.29, 0.12, 0.29

    #limits_imported = [DRY_P, WET_P, SNOW_P, FOG_P]
    limits = [0.25, 0.25, 0.25, 0.25]

    tot = sum(list(counter.values()))
    for i, k in enumerate(list(counter.keys())):
        res = round(counter[k]/tot, 2)
        if not (limits[i]-0.02<=res<=limits[i]+0.02):
            return False
    return True


def split(ratio, data, folder_info):
    print("Numero di annotation all'interno del csv: ", len(data))
    num_images = len(data)
    num_test_images = math.ceil(ratio*num_images)
    print("Train: {}, Test: {}".format(num_images-num_test_images, num_test_images))

    classes = ["dry", "wet", "snow", "fog"]
    counter = {c:0 for c in classes}
    done = False

    while not done:
        print("TEST")
        images_copy = deepcopy(data)
        for k in list(counter.keys()): counter[k]=0 # clear dict
        filenames = []
        for i in tqdm(range(num_test_images)):
            idx = random.randint(0, len(images_copy)-1)
            filename = images_copy[idx]
            filenames.append(filename)
            c = filename[1]
            counter[c] +=1
            images_copy.remove(filename)
        done = check_ds_distribution(counter)
        #print("TEST counter: ", counter)

        # if distribution is respected, process remaining images-annotations for training set
        if done:
            print("TRAIN")
            for k in list(counter.keys()): counter[k]=0 # clear dict
            for filename in tqdm(images_copy):
                c = filename[1]
                counter[c] +=1
            # print(counter)
            done = done and check_ds_distribution(counter)
            #print("TRAIN counter: ", counter)
    
    print("Split done!")
    print_filenames(os.path.join(folder_info, "test_filenames.txt"), filenames)
    print_filenames(os.path.join(folder_info, "train_filenames.txt"), images_copy)
    return filenames, images_copy


def adjust_read(folder, txt_file):
    with open(os.path.join(folder,txt_file),"r") as f:
        val_filenames = f.readlines()

    val_filenames = [f.rstrip() for f in val_filenames]
    new_filenames = []

    for f in val_filenames:
        path1, path2 = f.split(',')
        filename = [path1[2:-1],path2[2:-2]]
        new_filenames.append(filename)
    return new_filenames


def recover_filenames(folder:str)->list:
    # recover annotations for training and validation from file"
    val_filenames = adjust_read(folder, "test_filenames.txt")
    train_filenames = adjust_read(folder, "train_filenames.txt")

    return val_filenames, train_filenames


def move_images(val_filenames:list, train_filenames:list, dest:str):
    # starting from lists of filename, move annotations in the right folders #
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    #Creation folders
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        os.makedirs(os.path.join(train_dir, "dry"))
        os.makedirs(os.path.join(train_dir, "fog"))
        os.makedirs(os.path.join(train_dir, "wet"))
        os.makedirs(os.path.join(train_dir, "snow"))


    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        os.makedirs(os.path.join(test_dir, "dry"))
        os.makedirs(os.path.join(test_dir, "fog"))
        os.makedirs(os.path.join(test_dir, "wet"))
        os.makedirs(os.path.join(test_dir, "snow"))

    print("VAL")
    for f in tqdm(val_filenames):
        #shutil.copy(f[0], os.path.join(test_dir, f[1]))
        img = f[0].rsplit('/', 1)
        os.symlink(f[0], train_dir+"/"+f[1]+"/"+str(img[-1]))
    
    print("TRAIN")
    for f in tqdm(train_filenames):
        #shutil.copy(f[0], os.path.join(train_dir, f[1]))
        img = f[0].rsplit('/', 1)
        os.symlink(f[0], train_dir+"/"+f[1]+"/"+str(img[-1]))


if __name__ == '__main__':

    with open(CSV_PATH, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    folder = os.path.join(CODE_PATH,"partition_info")

    if not os.path.exists(folder):
        os.makedirs(folder)
        ratio = 0.1
        print("Create partition with split ratio {}".format(ratio))
        val_filenames, train_filenames = split(ratio, data, folder)

    else:
        val_filenames, train_filenames = recover_filenames(folder)
        print("Split recovered !")

    print("Moving symbolic links-image...")
    move_images(val_filenames, train_filenames, NEW_DATASET_PATH)
