import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

(x_train, y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train,x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test,y_test),
    epochs=10
)

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()

classes = list(range(10))

cm = confusion_matrix(y_test, model.predict(x_test).argmax(axis=1))

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
ticks = np.arange(len(classes))
plt.xticks(ticks, classes, rotation=-45)
plt.yticks(ticks, classes)
thresh = cm.max()/2.0
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,format(cm[i,j],'d'),
             horizontalalignment='center',
             color='white' if cm[i,j] > thresh else 'black')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()

test_loss, test_accuracy = model.evaluate(x_test,y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")