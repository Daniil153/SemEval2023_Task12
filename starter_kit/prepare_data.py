import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

languages = ['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo']

PROJECT_DIR = os.getcwd()
TASK = 'SubtaskA'
TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, TASK, 'train')
FORMATTED_TRAIN_DATA = os.path.join(TRAINING_DATA_DIR, 'formatted-train-data')

if os.path.isdir(TRAINING_DATA_DIR):
  print('Data directory found.')
  if not os.path.isdir(FORMATTED_TRAIN_DATA):
    print('Creating directory to store formatted data.')
    os.mkdir(FORMATTED_TRAIN_DATA)
else:
  print(TRAINING_DATA_DIR + ' is not a valid directory or does not exist!')

training_files = os.listdir(TRAINING_DATA_DIR)

if len(training_files) > 0:
    for training_file in training_files:
        if training_file.endswith('.tsv'):

            data = training_file.split('_')[0]
            if not os.path.isdir(os.path.join(FORMATTED_TRAIN_DATA, data)):
                print(data, 'Creating directory to store train, dev and test splits.')
                os.mkdir(os.path.join(FORMATTED_TRAIN_DATA, data))

            df = pd.read_csv(f'{TRAINING_DATA_DIR}/{training_file}', sep='\t', names=['ID', 'text', 'label'], header=0)
            df[['text', 'label']].to_csv(os.path.join(FORMATTED_TRAIN_DATA, data, 'train.tsv'), sep='\t', index=False)
        else:
            print(training_file + ' skipped!')
else:
    print('No files are found in this directory!')


if os.path.isdir(FORMATTED_TRAIN_DATA):
  print('Data directory found.')
  SPLITTED_DATA = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev')
  if not os.path.isdir(SPLITTED_DATA):
    print('Creating directory to store train, dev and test splits.')
    os.mkdir(SPLITTED_DATA)
else:
  print(FORMATTED_TRAIN_DATA + ' is not a valid directory or does not exist!')


formatted_training_files = os.listdir(FORMATTED_TRAIN_DATA)

if len(formatted_training_files) > 0:
  for data_name in formatted_training_files:
    formatted_training_file = os.path.join(data_name, 'train.tsv')
    if os.path.isfile(f"{FORMATTED_TRAIN_DATA}/{formatted_training_file}"):
      labeled_tweets = pd.read_csv(f"{FORMATTED_TRAIN_DATA}/{formatted_training_file}", sep='\t', names=['text', 'label'], header=0)
      train, dev = train_test_split(labeled_tweets, test_size=0.3)

      if not os.path.isdir(os.path.join(SPLITTED_DATA, data_name)):
        print(data_name, 'Creating directory to store train, dev and test splits.')
        os.mkdir(os.path.join(SPLITTED_DATA, data_name))

      train.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name, 'train.tsv'), sep='\t', index=False)
      dev.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name, 'dev.tsv'), sep='\t', index=False)
    else:
      print(training_file + ' is not a supported file!')
else:
  print('No files are found in this directory!')

