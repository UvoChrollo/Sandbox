# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = "https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json"
    urllib.request.urlretrieve(data_url, "sarcasm.json")

    f = open("sarcasm.json")
    data = json.load(f)
    f.close()

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = "post"
    padding_type = "post"
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    for item in data:
        sentences.append(item["headline"])
        labels.append(int(item["is_sarcastic"]))

    sentences = np.array(sentences)
    labels = np.array(labels)

    X_train, y_train = sentences[training_size:], labels[training_size:]
    X_test, y_test = sentences[:training_size], labels[:training_size]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    train_seq = tokenizer.texts_to_sequences(X_train)
    training_padded = pad_sequences(
        train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )

    test_seq = tokenizer.texts_to_sequences(X_test)
    testing_padded = pad_sequences(
        test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )

    # YOUR CODE HERE
    epochs = 15

    model = tf.keras.Sequential(
        [
            # YOUR CODE HERE. DO not change the last layer or test may fail
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, input_length=max_length
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ]
    )

    class sup_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("accuracy") > 0.80 and logs.get("val_accuracy") > 0.80:
                self.model.stop_training = True

    super_callback = sup_callback()

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(
        training_padded,
        y_train,
        epochs=epochs,
        validation_data=(testing_padded, y_test),
        verbose=2,
        callbacks=[super_callback],
    )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == "__main__":
    model = solution_C4()
    model.save("model_C4.h5")
