import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

vgg16 = VGG16(input_shape=[100,100,3], weights='imagenet', include_top=False)

x = Flatten()(vgg16.output)
x = Dense(32, activation='relu')(x)
x = Dense(4, activation='sigmoid')(x)

model = Model(vgg16.input, x)

def img_generator(no_of_batches=50, batch_size=32):
    while True:
        for _ in range(no_of_batches):
            X = np.zeros((batch_size, 100,100,3))
            Y = np.zeros((batch_size, 4))
            for i in range(batch_size):
                row0 = np.random.randint(90)
                col0 = np.random.randint(90)
                row1 = np.random.randint(row0,100)
                col1 = np.random.randint(col0,100)
                X[i,row0:row1,col0:col1,:] = 1
                Y[i,:] = [row0/100, col0/100, (row1-row0)/100, (col1-col0)/100]
            yield X,Y 

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.00001)
)

model.fit(
    img_generator(),
    steps_per_epoch=50,
    epochs=5
)

def make_prediction():
    x = np.zeros((100,100,3))
    row0 = np.random.randint(90)
    col0 = np.random.randint(90)
    row1 = np.random.randint(row0,100)
    col1 = np.random.randint(col0,100)
    x[row0:row1,col0:col1,:] = 1
    X = np.expand_dims(x,0)
    p = model.predict(X)[0]
    fig, ax = plt.subplots(1)
    ax.imshow(x)
    rect = Rectangle(
        (p[1]*100, p[0]*100),
        p[3]*100, p[2]*100, linewidth=1, edgecolor='r',facecolor='none'
    )
    ax.add_patch(rect)
    plt.show()

while True:
    make_prediction()
    key = input('Do you want to continue y/n: ')
    if key and key.lower() == 'n':
        break