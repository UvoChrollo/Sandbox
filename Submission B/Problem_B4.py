# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
import tensorflow as tf
import pandas as pd
import re
import numpy as np
import nltk

nltk.download("stopwords")

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords


def solution_B4():
    bbc = pd.read_csv(
        "https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv"
    )

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = "post"
    padding_type = "post"
    oov_tok = "<OOV>"
    training_portion = 0.8

    # YOUR CODE HERE
    def text_processor(data):
        for idx, val in enumerate(data):
            temp = re.sub(re.compile("<.*?>"), "", val.lower())
            pattern = re.compile(
                r"\b(" + r"|".join(stopwords.words("english")) + r")\b\s*"
            )
            data[idx] = pattern.sub("", temp)
            return data

    bbc["category"] = pd.factorize(bbc["category"])[0]

    # array
    data = np.array(bbc["text"])
    target = np.array(bbc["category"])

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, train_size=training_portion, random_state=42
    )

    X_train = text_processor(X_train)
    X_test = text_processor(X_test)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    train_pad = pad_sequences(
        X_train_seq, maxlen=max_length, truncating=trunc_type, padding=padding_type
    )
    test_pad = pad_sequences(
        X_test_seq, maxlen=max_length, truncating=trunc_type, padding=padding_type
    )

    class sup_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("accuracy") > 0.91 and logs.get("val_accuracy") > 0.91:
                self.model.stop_training = True

    super_callback = sup_callback()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size + 1, embedding_dim, input_length=max_length, trainable=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32)),
            tf.keras.layers.Dense(units=16, activation=tf.nn.relu),
            tf.keras.layers.Dense(6, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        train_pad,
        y_train,
        epochs=30,
        validation_data=(test_pad, y_test),
        callbacks=[super_callback],
    )
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.


if __name__ == "__main__":
    model = solution_B4()
    model.save("model_B4.h5")
