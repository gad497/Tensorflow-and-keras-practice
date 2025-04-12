import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

train_path = "Data/Food-5K/train"
test_path = "Data/Food-5K/test"
IMAGE_SIZE = [200,200]
folders = glob(train_path + '/*')
num_classes = len(folders)

ptm = VGG16(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False
)

ptm.trainable = False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

i = Input(shape=IMAGE_SIZE+[3])
x = preprocess_input(i)
x = data_augmentation(x)
x = ptm(x)
x = Flatten()(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(i,x)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    labels='inferred'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    labels='inferred'
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

plt.figure()
plt.title('Loss per epoch')
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()

plt.figure()
plt.title('Accuracy per epoch')
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend()

eval_path = 'Data/Food-5K/eval'
eval_ds = tf.keras.utils.image_dataset_from_directory(
    eval_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    labels='inferred'
)
class_names = eval_ds.class_names
eval_ds = eval_ds.cache().prefetch(buffer_size=AUTOTUNE)
predictions = model.predict(eval_ds)
y_pred = predictions.argmax(axis=1).astype(np.int32)
y_true=[]
for _,labels in eval_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)
cm = confusion_matrix(y_true, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Evaluation Data')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))