import tensorflow as tf 
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

checkpoint_filepath = 'ckpt/'
number_of_data = 107887  #put the correct number


def get_dataset_partitions_tf(ds, ds_size=number_of_data, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

def get_dataset_size(ds):
	count = 0
	for element in ds:
		count += 1
	return count

def normalize_inputs(ds):
	input_normalizer = layers.Normalization()
	input_normalizer.adapt(np.array(ds))
	return input_normalizer

def scheduler(epoch, lr):
	if epoch < 50:
		return float(lr)
	elif epoch%4 == 0:
		return float(lr * 0.5)
	else:
		return float(lr)

# Model No.1
def create_model(): #
	inputs = keras.Input(shape=(945,), name="past_data")
	x = layers.Dense(256, activation="relu", name="dense_1")(inputs)
	x = layers.Dense(128, activation="relu", name="dense_2")(x)
	outputs = layers.Dense(63,name="predictions")(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	return model

# # Model No.2
# def create_model():
# 	inputs = keras.Input(shape=(945,), name="past_data")
# 	x = layers.Dense(128, activation="relu", name="dense_1")(inputs)
# 	x = layers.Dense(128, activation="relu", name="dense_2")(x)
# 	outputs = layers.Dense(63,name="predictions")(x)
# 	model = keras.Model(inputs=inputs, outputs=outputs)
# 	return model

# # Model No.3
# def create_model():
# 	inputs = keras.Input(shape=(945,), name="past_data")
# 	x = layers.Dense(256, activation="relu", name="dense_1")(inputs)
# 	x = layers.Dense(128, activation="relu", name="dense_2")(x)
# 	x = layers.Dense(64, activation="relu", name="dense_3")(x)
# 	outputs = layers.Dense(63,name="predictions")(x)
# 	model = keras.Model(inputs=inputs, outputs=outputs)
# 	return model

dataset_path = "tfds_dataset/"
ds = tf.data.experimental.load("tfds_dataset/")
train_ds, val_ds, test_ds = get_dataset_partitions_tf(ds)

train_size = get_dataset_size(train_ds)
val_size = get_dataset_size(val_ds)
test_size = get_dataset_size(test_ds)

print(train_size,val_size,test_size)

train_ds = train_ds.shuffle(5000).batch(32)
val_ds = val_ds.shuffle(5000).batch(32)

model = create_model()
#model.load_weights(checkpoint_filepath)
model.summary()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='mean_absolute_error',metrics=["mae","acc"])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

history = model.fit(train_ds, batch_size=32, epochs=75, validation_data=val_ds, shuffle=True, callbacks=[model_checkpoint_callback,lr_callback])

