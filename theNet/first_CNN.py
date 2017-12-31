import os
from theNet.commands import train_neural_network, load_and_process_data, test

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os  # for doing directory operations
import numpy as np
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)

EPOCHS_COUNT = 40
VALIDATION_COUNT = 200 #hm patients will be in validation

labels_df = pd.read_csv('/home/talhassid/PycharmProjects/input/stage1_labels.csv', index_col=0)

data_dir = '/media/talhassid/My Passport/haimTal/stage1'
patients = os.listdir(data_dir)
#load_and_process_data(patients,labels_df,data_dir,"sample",train_flag=True)

data_dir_test_stage1 = '/media/talhassid/My Passport/haimTal/test/'
patients_test = os.listdir(data_dir_test_stage1)
#load_and_process_data(patients_test,labels_df,data_dir_test_stage1,"test_processed",train_flag=False)

data_dir_train = '/media/talhassid/My Passport/haimTal/train/'
patients_train = os.listdir(data_dir_train)
#load_and_process_data(patients_train,labels_df,data_dir_train,"train_processed",train_flag=True)


data_dir_test_stage2 = '/media/talhassid/My Passport/haimTal/stage2/stage2/'
patients_test = os.listdir(data_dir_test_stage2)
#load_and_process_data(patients_test,labels_df,data_dir_test_stage2,"test_processed",train_flag=False)

train_neural_network(epochs_count=EPOCHS_COUNT,validation_count=VALIDATION_COUNT)
test()

print ("finish")


