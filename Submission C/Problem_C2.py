# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


def solution_C2():
    mnist = tf.keras.datasets.mnist
    (train, label_tr), (test, label_te) = mnist.load_data()

    train = train / 255
    test = test / 255

    # YOUR CODE HERE
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(train, label_tr, validation_data=(test, label_te), epochs=5)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == "__main__":
    if __name__ == "__main__":
        model = solution_C2()
        model.save("model_C2.h5")
