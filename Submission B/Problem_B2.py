# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    # data
    (train_image, train_label), (test_image, test_label) = fashion_mnist.load_data()
    # rescale
    train_image = train_image / 255
    test_image = test_image / 255
    # modelling
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="softmax"),
        ]
    )
    # compile
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    # fit
    model.fit(
        train_image,
        train_label,
        validation_data=(test_image, test_label),
        epochs=10,
        verbose=1,
    )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == "__main__":
    model = solution_B2()
    model.save("model_B2.h5")
