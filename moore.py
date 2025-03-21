import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def schedule(epoch):
    if epoch > 50:
        return 0.0001
    return 0.001

data = pd.read_csv("Data/moore.csv", header=None).to_numpy()
X = data[:,0].reshape(-1,1)
Y = data[:,1]
X = X - X.mean()
# plt.scatter(X,Y)
Y = np.log(Y)
plt.figure()
plt.scatter(X,Y,label="actual")

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(0.001,0.9),
    loss='mse'
)

callback = tf.keras.callbacks.LearningRateScheduler(schedule)

model.fit(
    X,Y,
    epochs=200,
    callbacks=[callback]
)

model_weights = model.layers[0].get_weights()
# print(f"W: {model_weights[0][0,0]}, b: {model_weights[1][0]}")

y0 = model_weights[0][0,0]*X.min() + model_weights[1][0]
y1 = model_weights[0][0,0]*X.max() + model_weights[1][0]
y_points =  np.array([y0,y1])
x_points = np.array([X.min(),X.max()])
plt.plot(x_points, y_points, color='r', label="prediction")
plt.legend()
plt.title(f"W: {model_weights[0][0,0]}, b: {model_weights[1][0]}")
plt.show()