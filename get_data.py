import pandas as pd
import os
import time

# Setting variables
pwd = os.getcwd()
data_path = os.path.join(pwd, "data")
base_file_name = 'df_train_base.csv'
update_file_name = '_df_train_update.csv'

df_train = pd.read_csv("https://storage.googleapis.com/test-matteo/df_train_small.csv")

# Create data path
if not os.path.isdir(data_path):
  os.makedirs(data_path)

# Save initial data
if not os.path.isfile(data_path  + 'df_train_base.csv'):
    df_train = pd.read_csv('https://storage.googleapis.com/test-matteo/df_train_small.csv')
    df_train.to_csv(data_path + 'df_train_small.csv', index=False)
else:
    pass

while True:
  time_stamp = str(int(time.time()))
  df_train_update = pd.read_csv('https://creditrisk-cusvdt2goq-ez.a.run.app/training_update')
  df_train_update.to_csv(path_files + time_stamp + '_df_train_update.csv', index=False)
  print('SAVED ' + path_files + time_stamp + '_df_train_update.csv')
  time.sleep(55) # Delay for 55 seconds.


df_train.to_csv(data_path + 'df_train_base.csv', index=False)
