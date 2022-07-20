# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import nltk

nltk.download("stopwords")
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from nltk.corpus import stopwords


def solution_A4():
    def learning_rate(epochs):
        if epochs < 1:
            return 1e-3
        elif epochs < 3:
            return 1e-4
        elif epochs < 6:
            return 1e-5

    def text_processor(data):
        for idx, val in enumerate(data):
            _ = re.sub(re.compile("<.*?>"), "", val.lower())
            pattern = re.compile(
                r"\b(" + r"|".join(stopwords.words("english")) + r")\b\s*"
            )
            data[idx] = pattern.sub("", _)
        return data

    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb
    # YOUR CODE HERE
    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(s.numpy().decode("utf8"))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode("utf8"))
        testing_labels.append(l.numpy())

    # YOUR CODE HERE
    training_sentences = np.array(training_sentences)
    training_labels = np.array(training_labels)
    testing_sentences = np.array(testing_sentences)
    testing_labels = np.array(testing_labels)

    training_sentences = text_processor(training_sentences)
    testing_sentences = text_processor(testing_sentences)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = "post"
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    tokenizer.sequences_to_texts(training_sequences[0:1])
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    tokenizer.sequences_to_texts(testing_sequences[0:1])

    training_sentences_pad = pad_sequences(
        training_sequences, maxlen=max_length, truncating=trunc_type
    )
    testing_sentences_pad = pad_sequences(
        testing_sequences, maxlen=max_length, truncating=trunc_type
    )

    scheduler_learning_rate = tf.keras.callbacks.LearningRateScheduler(learning_rate)
    model = tf.keras.Sequential(
        [
            # YOUR CODE HERE. Do not change the last layer.
            tf.keras.layers.Embedding(
                vocab_size + 1, embedding_dim, input_length=max_length, trainable=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(14)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate(0)),
        metrics=["accuracy"],
    )

    model.fit(
        training_sentences_pad,
        training_labels,
        epochs=6,
        batch_size=16,
        validation_data=(testing_sentences_pad, testing_labels),
        callbacks=[scheduler_learning_rate],
    )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == "__main__":
    model = solution_A4()
    model.save("model_A4.h5")
