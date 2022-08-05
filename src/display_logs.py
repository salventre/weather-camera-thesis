import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging (1)
from plot_hist import plot_hist

'''
def adjsut_csv():
    path_in = r"E:\TESI\training_outputs\classifier\all_model_train_2\training_log_210.csv"
    path_out = r"E:\TESI\training_outputs\classifier\all_model_train_2\training_log_70.csv"
    start = 170

    with open(path_in, "r") as fin:
        with open(path_out, "a", newline='') as fout:
            reader = csv.reader(fin, delimiter = ',')
            writer = csv.writer(fout, delimiter = ',')
            reader.__next__() # skip header
            for row in reader:
                row[0]=int(row[0])+start
                writer.writerow(row)
'''

def plot():
    TRAIN_LOG_PATH = os.path.abspath('../weather-camera-thesis/data/log/training_log.csv')
    DOCS_PATH = os.path.abspath('../weather-camera-thesis/data/doc/')


    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)

    plot_hist(TRAIN_LOG_PATH, save=False, imgs_path=DOCS_PATH)

plot()