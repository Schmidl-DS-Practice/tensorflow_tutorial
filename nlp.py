import tensorflow as tf
import pandas as pd
import os
import re
import string
import nltk
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
keras = tf.keras
layers = keras.layers
preprocessing = keras.preprocessing
stopwords = nltk.corpus.stopwords
tokenizer = preprocessing.text.Tokenizer()
pad_sequences = preprocessing.sequence.pad_sequences()
nltk.download('stopwords')

def main():

    # https://www.kaggle.com/c/nlp-getting-started : NLP Disaster Tweets
    df = pd.read_csv("data/twitter_train.csv")

    print((df.target == 1).sum()) # Disaster
    print((df.target == 0).sum()) # No Disaster

    def remove_URL(text):
        url = re.compile(r"https?://\S+|www\.\S+")
        return url.sub(r"", text)

    # https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022
    def remove_punct(text):
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    pattern = re.compile(r"https?://(\S+|www)\.\S+")
    for t in df.text:
        matches = pattern.findall(t)
        for match in matches:
            print(t)
            print(match)
            print(pattern.sub(r"", t))
        if len(matches) > 0:
            break

    df["text"] = df.text.map(remove_URL) # map(lambda x: remove_URL(x))
    df["text"] = df.text.map(remove_punct)

    # remove stopwords
    # Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine
    # has been programmed to ignore, both when indexing entries for searching and when retrieving them
    # as the result of a search query.
    stop = set(stopwords.words("english"))

    # https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
    def remove_stopwords(text):
        filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(filtered_words)

    df["text"] = df.text.map(remove_stopwords)

    # Split dataset into training and validation set
    train_size = int(df.shape[0] * 0.8)

    train_df = df[:train_size]
    val_df = df[train_size:]

    # split text and labels
    train_sentences = train_df.text.to_numpy()
    train_labels = train_df.target.to_numpy()
    val_sentences = val_df.text.to_numpy()
    val_labels = val_df.target.to_numpy()

    train_sentences.shape, val_sentences.shape

    # Tokenize
    # vectorize a text corpus by turning each text into a sequence of integers
    tokenizer = Tokenizer(num_words=num_unique_words)
    tokenizer.fit_on_texts(train_sentences) # fit only to training

    # each word has unique index
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)

    print(train_sentences[10:15])
    print(train_sequences[10:15])

    # Pad the sequences to have the same length
    # Max number of words in a sequence
    max_length = 20

    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
    val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")

    print(train_sentences[10])
    print(train_sequences[10])
    print(train_padded[10])

    # Check reversing the indices
    # flip (key, value)
    reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])

    def decode(sequence):
        return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])

    decoded_text = decode(train_sequences[10])

    print(train_sequences[10])
    print(decoded_text)

    # Create LSTM model
      # Embedding: https://www.tensorflow.org/tutorials/text/word_embeddings
    # Turns positive integers (indexes) into dense vectors of fixed size. (other approach could be one-hot-encoding)

    # Word embeddings give us a way to use an efficient, dense representation in which similar words have
    # a similar encoding. Importantly, you do not have to specify this encoding by hand. An embedding is a
    # dense vector of floating point values (the length of the vector is a parameter you specify).

    model = keras.models.Sequential()
    model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))

    # The layer will take as input an integer matrix of size (batch, input_length),
    # and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
    # Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.


    model.add(layers.LSTM(64, dropout=0.1))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.summary()

    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(lr=0.001)
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    model.fit(train_padded, train_labels, epochs=20, validation_data=(val_padded, val_labels), verbose=2)

    predictions = model.predict(train_padded)
    predictions = [1 if p > 0.5 else 0 for p in predictions]

    print(train_sentences[10:20])

    print(train_labels[10:20])
    print(predictions[10:20])

if __name__ == "__main__":
    main()