import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import config as cfg

class ChatbotModel(Model):
  def __init__(self, weights_file=None, embedding_matrix=None):
    super(ChatbotModel, self).__init__()

    # Define the embedding layer, shared by the decoder and the encoder
    embedding = None
    if weights_file == None:
        embedding = layers.Embedding(output_dim=cfg.WORD_EMBEDDING_SIZE, input_dim=cfg.VOCABULARY_SIZE, input_length=cfg.MAX_SEQUENCE_LENGTH, weights=[embedding_matrix])
    else:
        embedding = layers.Embedding(output_dim=cfg.WORD_EMBEDDING_SIZE, input_dim=cfg.VOCABULARY_SIZE, input_length=cfg.MAX_SEQUENCE_LENGTH)
        
    # Input 1: Context (question)
    self.input_context = layers.Input(shape=(cfg.MAX_SEQUENCE_LENGTH,), dtype="int32")
    self.embedding_context = embedding
    # LSTM encoder
    self.lstm_context = layers.LSTM(cfg.SENTENCE_EMBEDDING_SIZE, kernel_initializer="lecun_uniform") 

    # Input 2: Answer
    self.input_answer = layers.Input(shape=(cfg.MAX_SEQUENCE_LENGTH,), dtype="int32")
    self.embedding_answer = embedding
    # LSTM decoder
    self.lstm_answer = layers.LSTM(cfg.SENTENCE_EMBEDDING_SIZE, kernel_initializer="lecun_uniform")
    
    # Merge the inputs
    self.merge_layer = layers.Concatenate(axis=1)
    
    # Fully-connected layers
    self.dense1 = layers.Dense(cfg.VOCABULARY_SIZE // 2, activation="relu")
    #self.batchnorm = layers.BatchNormalization()
    self.dense2 = layers.Dense(cfg.VOCABULARY_SIZE, activation="softmax")
    
    self.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizers.Adam(lr=cfg.LEARNING_RATE),
        metrics=["accuracy"]
    )

    if weights_file != None:
        self.predict([np.zeros((1, cfg.MAX_SEQUENCE_LENGTH)), np.zeros((1, cfg.MAX_SEQUENCE_LENGTH))])
        self.load_weights(weights_file)


  def call(self, inputs):
    x_context = self.embedding_context(inputs[0])
    x_context = self.lstm_context(x_context)

    x_answer = self.embedding_answer(inputs[1])
    x_answer = self.lstm_answer(x_answer)

    merged = self.merge_layer([x_context, x_answer])
    out = self.dense1(merged)
    #out = self.batchnorm(out)
    out = self.dense2(out)

    return out
