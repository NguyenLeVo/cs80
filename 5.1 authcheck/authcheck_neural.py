# Using neural networks to determine the authenticity of a coin

import csv
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Read data from file
with open("banknotes.csv") as file:
    reader = csv.reader(file)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row [4] == "0" else 0
        })

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
label = [row["label"] for row in data]
X_train, X_test, y_train, y_test = train_test_split(
    evidence, label, test_size=0.4
)

# Create a neural network
model = tf.keras.models.Sequential()

# Add a hidden layer with 8 units, with ReLU activation function
model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"))

# Add output layer with 1 unit, with sigmoid activation
# 1 unit because either real or fake
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Train neural network
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_train, y_train, epochs=20)

# Evaluate how well model performs
model.evaluate(X_test, y_test, verbose=2)

'''
Epoch 1/20
26/26 [==============================] - 0s 1ms/step - loss: 0.4747 - accuracy: 0.7789
Epoch 2/20
26/26 [==============================] - 0s 1ms/step - loss: 0.4149 - accuracy: 0.8202
Epoch 3/20
26/26 [==============================] - 0s 1ms/step - loss: 0.3701 - accuracy: 0.8578
Epoch 4/20
26/26 [==============================] - 0s 1ms/step - loss: 0.3362 - accuracy: 0.8712
Epoch 5/20
26/26 [==============================] - 0s 1ms/step - loss: 0.3084 - accuracy: 0.8773
Epoch 6/20
26/26 [==============================] - 0s 1ms/step - loss: 0.2847 - accuracy: 0.8834
Epoch 7/20
26/26 [==============================] - 0s 1ms/step - loss: 0.2643 - accuracy: 0.8906
Epoch 8/20
26/26 [==============================] - 0s 1ms/step - loss: 0.2458 - accuracy: 0.9016
Epoch 9/20
26/26 [==============================] - 0s 1ms/step - loss: 0.2291 - accuracy: 0.9101
Epoch 10/20
26/26 [==============================] - 0s 997us/step - loss: 0.2142 - accuracy: 0.9222
Epoch 11/20
26/26 [==============================] - 0s 1ms/step - loss: 0.2004 - accuracy: 0.9307
Epoch 12/20
26/26 [==============================] - 0s 959us/step - loss: 0.1878 - accuracy: 0.9368
Epoch 13/20
26/26 [==============================] - 0s 960us/step - loss: 0.1755 - accuracy: 0.9405
Epoch 14/20
26/26 [==============================] - 0s 960us/step - loss: 0.1646 - accuracy: 0.9490
Epoch 15/20
26/26 [==============================] - 0s 1ms/step - loss: 0.1540 - accuracy: 0.9526
Epoch 16/20
26/26 [==============================] - 0s 1ms/step - loss: 0.1447 - accuracy: 0.9550
Epoch 17/20
26/26 [==============================] - 0s 960us/step - loss: 0.1357 - accuracy: 0.9599
Epoch 18/20
26/26 [==============================] - 0s 1ms/step - loss: 0.1278 - accuracy: 0.9648
Epoch 19/20
26/26 [==============================] - 0s 844us/step - loss: 0.1207 - accuracy: 0.9684
Epoch 20/20
26/26 [==============================] - 0s 997us/step - loss: 0.1141 - accuracy: 0.9745
18/18 - 0s - loss: 0.1195 - accuracy: 0.9672
'''