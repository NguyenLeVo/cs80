# Using neural network to recognize handwriting digits

import sys
import tensorflow as tf

# Use MNIST handwriting dataset
mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

# Create a convolutional neural network
model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),

    # Max-pooling later, using 2x2 pool size
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    # Flatten units
    tf.keras.layers.Flatten(),

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(10, activation="softmax")
])

# Train neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

# Evaluate neural netwok performance
model.evaluate(x_test, y_test, verbose=2)

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")

'''
Epoch 1/10
1875/1875 [==============================] - 16s 9ms/step - loss: 0.2493 - accuracy: 0.9244
Epoch 2/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.1069 - accuracy: 0.9685
Epoch 3/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0816 - accuracy: 0.9758
Epoch 4/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0650 - accuracy: 0.9803
Epoch 5/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0559 - accuracy: 0.9829
Epoch 6/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0496 - accuracy: 0.9849
Epoch 7/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0416 - accuracy: 0.9870
Epoch 8/10
1875/1875 [==============================] - 15s 8ms/step - loss: 0.0360 - accuracy: 0.9879
Epoch 9/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0326 - accuracy: 0.9888
Epoch 10/10
1875/1875 [==============================] - 16s 8ms/step - loss: 0.0318 - accuracy: 0.9893
313/313 - 1s - loss: 0.0358 - accuracy: 0.9897
Model saved to model.h5.
'''