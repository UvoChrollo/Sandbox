# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback


def solution_B3():
    data_url = "https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip"
    urllib.request.urlretrieve(data_url, "rps.zip")
    local_file = "rps.zip"
    zip_ref = zipfile.ZipFile(local_file, "r")
    zip_ref.extractall("data/")
    zip_ref.close()

    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.4,
    )

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode="categorical",
        subset="training",
    )

    test_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode="categorical",
        subset="validation",
    )

    class sup_callback(Callback):
        def on_epoch_end(self, epoch, logs={}):
            """
            Halts the training after reaching 60 percent accuracy

            Args:
              epoch (integer) - index of epoch (required but unused in the function definition below)
              logs (dict) - metric results from the training epoch
            """

            # Check accuracy
            if logs.get("accuracy") > 0.90 and logs.get("val_accuracy") > 0.90:

                # Stop if threshold is met
                print("\naccuracy and validation accuracy is greater than 0.90")
                self.model.stop_training = True

    # Instantiate class
    sup_cb = sup_callback()

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(150, 150, 3)
            ),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(units=3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=15,
        verbose=1,
        callbacks=[sup_cb],
    )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == "__main__":
    model = solution_B3()
    model.save("model_B3.h5")
