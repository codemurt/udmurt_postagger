import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle

class UdmurtPOSTagger:
    @staticmethod
    def _ignore_accuracy():
        import tensorflow as tf
        def ignore_accuracy(y_true, y_pred):
            y_true_class = tf.argmax(y_true, axis=-1)
            y_pred_class = tf.argmax(y_pred, axis=-1)
            ignore_mask = tf.cast(tf.not_equal(y_true_class, 0), 'int32')
            matches = tf.cast(tf.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = tf.reduce_sum(matches) / tf.maximum(tf.reduce_sum(ignore_mask), 1)
            return accuracy
        return ignore_accuracy

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resources_dir = os.path.join(current_dir, 'resources')
        
        # Load model
        model_path = os.path.join(resources_dir, 'udmurt_pos_tagger_model.h5')
        self.model = load_model(
            model_path,
            custom_objects={'ignore_accuracy': self._ignore_accuracy()}
        )
        
        # Load dictionaries
        with open(os.path.join(resources_dir, 'udmurt_pos_word2index.pkl'), 'rb') as f:
            self.word2index = pickle.load(f)
        
        with open(os.path.join(resources_dir, 'udmurt_pos_index2tag.pkl'), 'rb') as f:
            self.index2tag = pickle.load(f)
        
        # Get max sequence length from model
        self.max_length = self.model.input_shape[1]
    
    def predict(self, tokens):
        # Convert tokens to indices
        sequence = []
        for token in tokens:
            token_lower = token.lower()
            sequence.append(self.word2index.get(token_lower, self.word2index['-OOV-']))
        
        # Apply padding
        padded_sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
        
        # Predict tags
        prediction = self.model.predict(padded_sequence)
        
        # Convert predictions to tags
        tag_indices = prediction[0][:len(tokens)].argmax(axis=-1)
        tags = [self.index2tag[idx] for idx in tag_indices]
        
        return tags