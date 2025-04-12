# Use features as input to train network

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


train_path = "Data/Food-5K/train"
test_path = "Data/Food-5K/test"
IMAGE_SIZE = [200,200]
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

def preprocess(image,label):
    return preprocess_input(image),label

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

ptm = VGG16(
    input_shape=IMAGE_SIZE+[3],
    weights='imagenet',
    include_top=False
)

x = Flatten()(ptm.output)
model = Model(inputs=ptm.input, outputs=x)

X_train = model.predict(train_ds)
X_valid = model.predict(val_ds)

y_train = np.concatenate([y for x,y in train_ds])
y_valid = np.concatenate([y for x,y in val_ds])

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_valid = scalar.fit_transform(X_valid)

feat = model.predict(np.random.random([1]+IMAGE_SIZE+[3]))
in_shape = feat.shape[1]

i = Input(shape=in_shape)
x = Dense(1, activation='sigmoid')(i)
linear_model = Model(inputs=i, outputs=x)

linear_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = linear_model.fit(
    X_train,y_train,
    validation_data=(X_valid,y_valid),
    epochs=5
)

plt.figure()
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.title('loss per epoch')

plt.figure()
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.legend()
plt.title('accuracy per epoch')
plt.show()
