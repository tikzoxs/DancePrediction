import pandas as pd
import numpy as np
import re
import csv
import tensorflow as tf


new_dataset = tf.data.experimental.load("tfds_dataset/")
print(new_dataset)
count = 0
for element in new_dataset:
	count += 1
print(count)