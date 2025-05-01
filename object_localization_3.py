from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from matplotlib.patches import Rectangle

char = image.load_img('Data/charmander-tight.png')
# plt.imshow(char)
# plt.show()

img_dim = 200
char_h, char_w, _ = np.array(char).shape

def img_gen(no_of_batches, batch_size):
    while True:
        for _ in range(no_of_batches):
            X = np.zeros((batch_size, img_dim, img_dim, 3))
            Y = np.zeros((batch_size,4))
            for i in range(batch_size):
                scale = np.random.random() + 0.5
                scaled_char_h, scaled_char_w = int(scale*char_h), int(scale*char_w)
                resized_img = resize(np.array(char), (scaled_char_h, scaled_char_w), preserve_range=True).astype(np.uint8)
                row0 = np.random.randint(200-scaled_char_h)
                col0 = np.random.randint(200-scaled_char_w)
                row1 = row0 + scaled_char_h
                col1 = col0 + scaled_char_w
                X[i, row0:row1, col0:col1, :] = resized_img
                Y[i,:] = np.array([row0/img_dim, col0/img_dim, scaled_char_h/img_dim, scaled_char_w/img_dim])
            yield X/255.0,Y

vgg = VGG16(input_shape=[200,200,3], weights='imagenet', include_top=False)
# vgg.trainable = False
x = Flatten()(vgg.output)
x = Dense(4, activation='sigmoid')(x)
model = Model(vgg.input, x)

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.0001)
)

model.fit(
    img_gen(50,32),
    steps_per_epoch=50,
    epochs=5
)

def plot_predictions():
    X = np.zeros((img_dim, img_dim, 3))
    scale = np.random.random() + 0.5
    scaled_char_h, scaled_char_w = int(scale*char_h), int(scale*char_w)
    resized_img = resize(np.array(char), (scaled_char_h, scaled_char_w), preserve_range=True).astype(np.uint8)
    row0 = np.random.randint(200-scaled_char_h)
    col0 = np.random.randint(200-scaled_char_w)
    row1 = row0 + scaled_char_h
    col1 = col0 + scaled_char_w
    X[row0:row1, col0:col1, :] = resized_img
    p = model.predict(np.expand_dims(X,0)/255.0)[0]
    fig, axes = plt.subplots(1)
    axes.imshow(X.astype(np.uint8))
    rect = Rectangle((p[1]*img_dim, p[0]*img_dim),
                     p[3]*img_dim, p[2]*img_dim, linewidth=1, edgecolor='r', facecolor='none')
    axes.add_patch(rect)
    rect = Rectangle((col0, row0),
                     scaled_char_w, scaled_char_h, linewidth=1, edgecolor='g', facecolor='none')
    axes.add_patch(rect)
    plt.show()

while True:
    plot_predictions()
    key = input('Continue y/n: ')
    if key and key.lower() == 'n':
        break