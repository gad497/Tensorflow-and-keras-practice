from glob import glob
import imageio.v3 as iio
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from matplotlib.patches import Rectangle

img_dim = 200
char = iio.imread('Data/charmander-tight.png')
char_h, char_w, _ = char.shape

bg_list = glob('Data/backgrounds/*.jpg')


def img_gen(batch_size):
    while True:
        X = np.zeros((batch_size, img_dim, img_dim, 3))
        Y = np.zeros((batch_size, 4))
        for i in range(batch_size):
            bg = np.random.choice(bg_list) # choose random background 
            bg_img = iio.imread(bg)
            bg_h, bg_w, _ = bg_img.shape
            bg_row0 = np.random.randint(bg_h-img_dim)
            bg_col0 = np.random.randint(bg_w-img_dim)
            bg_row1 = bg_row0 + img_dim
            bg_col1 = bg_col0 + img_dim
            X[i] = bg_img[bg_row0:bg_row1, bg_col0:bg_col1, :] # resize to img_dim and add to batch
            scale = np.random.random() + 0.5
            fg_img = resize(char, (scale*char_h, scale*char_w), preserve_range=True) # scale foreground image
            if np.random.random() < 0.5:
                fg_img = np.fliplr(fg_img) # flip foreground image randomly
            mask = (fg_img[:,:,3] == 0) # boolean mask
            fg_h, fg_w, _ = fg_img.shape
            row0 = np.random.randint(img_dim-fg_h)
            col0 = np.random.randint(img_dim-fg_w)
            row1 = row0 + fg_h
            col1 = col0 + fg_w
            X[i, row0:row1, col0:col1, :] = (X[i, row0:row1, col0:col1, :] * np.expand_dims(mask,-1)) + fg_img[:,:,:3]
            Y[i,:] = np.array([row0/img_dim, col0/img_dim, fg_h/img_dim, fg_w/img_dim])
        yield X/255.0, Y

vgg = VGG16(input_shape=[200,200,3], weights='imagenet', include_top=False)
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
    bg = np.random.choice(bg_list)
    bg_img = iio.imread(bg)
    bg_h, bg_w, _ = bg_img.shape
    bg_row0 = np.random.randint(bg_h-img_dim)
    bg_col0 = np.random.randint(bg_w-img_dim)
    bg_row1 = bg_row0 + img_dim
    bg_col1 = bg_col0 + img_dim
    X = bg_img[bg_row0:bg_row1, bg_col0:bg_col1, :]
    scale = np.random.random() + 0.5
    fg_img = resize(char, (scale*char_h, scale*char_w), preserve_range=True)
    if np.random.random() < 0.5:
        fg_img = np.fliplr(fg_img)
    mask = (fg_img[:,:,3] == 0)
    fg_h, fg_w, _ = fg_img.shape
    row0 = np.random.randint(img_dim-fg_h)
    col0 = np.random.randint(img_dim-fg_w)
    row1 = row0 + fg_h
    col1 = col0 + fg_w
    X[row0:row1, col0:col1, :] = (X[row0:row1, col0:col1, :] * np.expand_dims(mask,-1)) + fg_img[:,:,:3]
    p = model.predict(np.expand_dims(X/255.0,0))[0]
    _ , ax = plt.subplots(1)
    rect = Rectangle((p[1]*img_dim, p[0]*img_dim),
                     p[3]*img_dim, p[2]*img_dim, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.imshow(X)
    plt.show()

while True:
    plot_prediction()
    key = input("Continue y/n: ")
    if key and key.lower() == 'n':
        break