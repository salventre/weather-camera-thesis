from telegram.ext import Updater, CommandHandler
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath("/home/weather-camera-thesis/src"))
from plot_hist import plot_hist

TOKEN="5528518099:AAHhPoPkCxt4-Kf6W_ed7oqkq52M1OQ4M54" #to check

n_experiment = "1_experiment" #specify the expriment to visualize
CSV_PATH_COLAB = "/content/drive/Shareddrives/WeatherCamera/Project Saves/{}/logs/training_log.csv".format(n_experiment)
IMGS_PATH_COLAB = "/content/drive/Shareddrives/WeatherCamera/Project Saves/{}/doc".format(n_experiment)

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

def stats(update, context):
    try:
        df = pd.read_csv(CSV_PATH_COLAB)

        msg = ""
        msg += n_experiment + "\n"
        msg += 'Epoch: ' + str(len(df)) + "\n\n"
        msg += "Best training loss: " + str(round(min(df['loss']), 3)) + "\n"
        msg += "Best validation loss: " + str(round(min(df['val_loss']), 3)) + "\n\n"
        msg += "Best training accuracy: " + str(round(max(df['categorical_accuracy']), 3)) + "\n"
        msg += "Best validation accuracy: " + str(round(max(df['val_categorical_accuracy']), 3)) + "\n\n"
        msg += "Last training loss: " + str(round(df['loss'].iloc[-1], 3)) + "\n"
        msg += "Last validation loss: " + str(round(df['val_loss'].iloc[-1], 3)) + "\n\n"
        msg += "Last training accuracy: " + str(round(df['categorical_accuracy'].iloc[-1], 3)) + "\n"
        msg += "Last validation accuracy: " + str(round(df['val_categorical_accuracy'].iloc[-1], 3)) + "\n\n"

        context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except:
        context.bot.send_message(chat_id=update.effective_chat.id, text="No info founded")

def update(update, context):
    try:
        path_loss_fig, path_acc_fig = plot_hist(CSV_PATH_COLAB, save=True, imgs_path=IMGS_PATH_COLAB)
        context.bot.sendPhoto(chat_id=update.effective_chat.id, photo=open(path_loss_fig, 'rb'), caption="Loss")
        context.bot.sendPhoto(chat_id=update.effective_chat.id, photo=open(path_acc_fig, 'rb'), caption="accuracy")
    except:
        context.bot.send_message(chat_id=update.effective_chat.id, text="No info founded")

def main():
   upd= Updater(TOKEN, use_context=True)
   disp=upd.dispatcher
   disp.add_handler(CommandHandler("update", update))
   disp.add_handler(CommandHandler("stats", stats))
   disp.add_handler(CommandHandler("start", start))
   upd.start_polling()
   upd.idle()

if __name__=='__main__':
   main()
