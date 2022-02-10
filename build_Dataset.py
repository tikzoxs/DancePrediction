import pandas as pd
import numpy as np
import re
import csv
import tensorflow as tf


dance_data = pd.read_csv("data.csv") #path to the csv
source_df = dance_data.copy()
target_df = source_df.pop('target')

source_list = []
target_list = []
for s_row, t_row in zip(source_df.values, target_df.values):
	ss = re.sub('[\[\]]+', '', s_row[0]).split(', ')
	float_ss = list(map(float, ss))
	tt = re.sub('[\[\]]+', '', t_row).split(', ')
	float_tt = list(map(float, tt))
	source_list.append(float_ss)
	target_list.append(float_tt)


dataset = tf.data.Dataset.from_tensor_slices((source_list,target_list))
tf.data.experimental.save(dataset, "tfds_dataset/") #create this folder before running

\

