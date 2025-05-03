from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

img = np.array(image.load_img('Data/charmander-tight.png'))
img_h, img_w, _ = img.shape
img_dim = 200

def img_gen(batch_size):
    while True:
        X = np.zeros((batch_size, img_dim, img_dim, 3))
        Y = np.zeros((batch_size, 4))
        for i in range(batch_size):
            row0 = np.random.randint(img_dim-img_h)
            col0 = np.random.randint(img_dim-img_w)
            row1 = row0 + img_h
            col1 = col0 + img_w
            if np.random.random() < 0.5:
                X[i, row0:row1, col0:col1, :] = np.fliplr(img)
            else:
                X[i, row0:row1, col0:col1, :] = img
            Y[i,:] = np.array([row0/img_dim, col0/img_dim, img_h/img_dim, img_w/img_dim])
        yield X/255.0, Y

vgg = VGG16(input_shape=[img_dim, img_dim, 3], weights='imagenet', include_top=False)
x = Flatten()(vgg.output)
x = Dense(4, activation='sigmoid')(x)
model = Model(vgg.input, x)

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.0001)
)

model.fit(
    img_gen(32),
    steps_per_epoch=50,
    epochs=3
)

def plot_prediction():
    X = np.zeros((img_dim, img_dim, 3))
    row0 = np.random.randint(img_dim-img_h)
    col0 = np.random.randint(img_dim-img_w)
    row1 = row0 + img_h
    col1 = col0 + img_w
    if np.random.random() < 0.5:
        X[row0:row1, col0:col1, :] = np.fliplr(img)
    else:
        X[row0:row1, col0:col1, :] = img
    p = model.predict(np.expand_dims(X,0)/255.0)[0]
    _ , axes = plt.subplots(1)
    prediction = Rectangle((p[1]*img_dim, p[0]*img_dim),
                     p[3]*img_dim,p[2]*img_dim,linewidth=1,edgecolor='r',facecolor='none')
    axes.add_patch(prediction)
    actual = Rectangle((col0, row0),
                       img_w, img_h,linewidth=1,edgecolor='g',facecolor='none')
    axes.add_patch(actual)
    plt.imshow(X.astype(np.uint8))
    plt.show()

while True:
    plot_prediction()
    key = input("Continue y/n: ")
    if key and key.lower() == 'n':
        break