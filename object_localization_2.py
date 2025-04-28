from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from matplotlib.patches import Rectangle

char = image.load_img('Data/charmander-tight.png')
np_char = np.array(char)
plt.imshow(char)
plt.title(np_char.shape)
plt.show()

img_dim = 200
char_h, char_w, _ = np.array(char).shape

def img_generator(no_of_batches, batch_size, img_dim):
    while True:
        for _ in range(no_of_batches):
            X = np.zeros((batch_size, img_dim, img_dim, 3))
            Y = np.zeros((batch_size, 4))
            for i in range(batch_size):
                row0 = np.random.randint(200-char_h)
                col0 = np.random.randint(200-char_w)
                row1 = row0 + char_h
                col1 = col0 + char_w
                X[i,row0:row1, col0:col1, :] = np_char
                Y[i,:] = np.array([row0/img_dim, col0/img_dim, char_h/img_dim, char_w/img_dim])
        yield X/255.0,Y

def make_model(img_dim):
    vgg = VGG16(input_shape=[img_dim, img_dim, 3], weights='imagenet', include_top=False)
    x = Flatten()(vgg.output)
    x = Dense(4, activation='sigmoid')(x)
    return Model(vgg.input, x)

model = make_model(img_dim)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001)
)

model.fit(
    img_generator(50, 32, 200),
    steps_per_epoch=50,
    epochs=5
)

def plot_predictions():
    X = np.zeros((img_dim, img_dim, 3))
    row0 = np.random.randint(200-char_h)
    col0 = np.random.randint(200-char_w)
    row1 = row0 + char_h
    col1 = col0 + char_w
    X[row0:row1, col0:col1, :] = np_char
    p = model.predict(np.expand_dims(X,0)/255.0)[0]

    fig, axes = plt.subplots(1)
    axes.imshow(X.astype(np.uint8))
    rect = Rectangle((col0, row0), char_w, char_h, linewidth=1, edgecolor='g', facecolor='none')
    axes.add_patch(rect)
    rect = Rectangle((p[1]*img_dim, p[0]*img_dim),
                     p[3]*img_dim, p[2]*img_dim, linewidth=1, edgecolor='r', facecolor='none')
    axes.add_patch(rect)
    plt.show()

while True:
    plot_predictions()
    key = input("Continue? y/n: ")
    if key and key.lower() == 'n':
        break