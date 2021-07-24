import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, test_text):
        self.model = tf.keras.models.load_model("../model.h5")
        self.test_text = test_text
        self.word_index = tf.keras.datasets.imdb.get_word_index()
        self._preprocessing()
        
    def _preprocessing(self):
        self.word_index = {k:(v+3) for k,v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        self.word_index["<UNUSED>"] = 3

    def _encode_review(self):
        arr = tf.keras.preprocessing.text.text_to_word_sequence(self.test_text)    
        res = []
        for k, v in enumerate(arr):
            try: res.append(self.word_index[v])
            except KeyError: res.append(self.word_index["<UNK>"])
        return np.array(res)
    
    def _padding(self):
        x = [self._encode_review()]
        x = tf.keras.preprocessing.sequence.pad_sequences(
                x,
                value=self.word_index["<PAD>"],
                padding="post",
                maxlen=512
            )
        return x
    
    def predict(self):
        result = self.model.predict(self._padding())
        return ("positive", result[0][0]) if result > 0.5 else ("negative", result[0][0])
