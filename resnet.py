import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn

train_path = 'Data/fruits-360/Training'
val_path = 'Data/fruits-360/Validation'
IMAGE_SIZE = [100, 100]
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.2,0.2)
])

ptm = ResNet50(
    input_shape=IMAGE_SIZE+[3],
    weights='imagenet',
    include_top=False
)

ptm.trainable = False

i = Input(shape=IMAGE_SIZE+[3])
x = preprocess_input(i)
x = data_augmentation(x)
x = ptm(x)
x = Flatten()(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(i,x)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

plt.figure()
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('loss per epoch')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.title('accuracy per epoch')
plt.legend()

predictions = model.predict(val_ds)
y_pred = predictions.argmax(axis=1).astype('int')
y_true = []
for _, label in val_ds:
    y_true.extend(label.numpy())
y_true = np.array(y_true)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15,10))
seaborn.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
plt.xlabel('Predictions')
plt.ylabel('True label')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

print(classification_report(y_true,y_pred,target_names=class_names))