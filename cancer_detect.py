from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
import tensorflow as tf

data = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(data.data, data.target, test_size=0.2)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(30,)),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
r = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=100
)

final_loss, final_acc = model.evaluate(X_test,y_test)
print(f"Final loss: {final_loss:.4f}, Final accuracy: {final_acc*100:.2f}%")