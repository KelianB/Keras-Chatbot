import numpy as np
import gc
from sklearn.model_selection import train_test_split
#from keras.callbacks import EarlyStopping

import config as cfg
import utils
from model import ChatbotModel

EPOCHS = 30
BATCH_SIZE = 512
num_subsets = 10
n_val = 2000

weights_file = "model_weights.h5"

# If true, the model will be initialized with the weights from the line above.
resume_training = False

"""
def get_response(model, inp, index_to_word):
    ans_partial = np.zeros((1,cfg.MAX_SEQUENCE_LENGTH))
    ans_partial[0, -1] = cfg.BOS_VOCAB_INDEX  #  the index of the symbol BOS (begin of sentence)
    for k in range(cfg.MAX_SEQUENCE_LENGTH - 1):
        # Predict the next word
        pred_one_hot = model.predict([inp, ans_partial])
        word_index = np.argmax(pred_one_hot)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = word_index

    seq = [k.astype(int) for k in ans_partial[0]]
    return utils.seq_to_text(seq, index_to_word)
"""


def prepare_fit_data(q_slice, a_slice):
  count_words = 0
  for i, sent in enumerate(a_slice):
      limit = np.where(sent==cfg.EOS_VOCAB_INDEX)[0][0] #  the position of the symbol EOS
      count_words += limit

  Q = np.zeros((count_words,cfg.MAX_SEQUENCE_LENGTH), dtype="int16")
  A = np.zeros((count_words,cfg.MAX_SEQUENCE_LENGTH), dtype="int16")
  Y = np.zeros((count_words,cfg.VOCABULARY_SIZE), dtype="int16")

  # Loop over the training examples:
  word_index = 0
  for i, sent in enumerate(a_slice):
      ans_partial = np.zeros((1,cfg.MAX_SEQUENCE_LENGTH))
      
      limit = np.where(sent==cfg.EOS_VOCAB_INDEX)[0][0]  #  the position of the symbol EOS

      # Iterate over the words of the current target output (the current output sequence):
      for k in range(1,limit+1):
          # One-hot encoding
          y = np.zeros((1, cfg.VOCABULARY_SIZE))
          y[0, sent[k]] = 1

          # Prepare partial answer to input
          ans_partial[0,-k:] = sent[0:k]

          # training the model for one epoch using teacher forcing:

          Q[word_index, :] = q_slice[i:i+1] 
          A[word_index, :] = ans_partial 
          Y[word_index, :] = y
          word_index += 1

  return Q,A,Y


def train():
    # Get the vocabulary
    index_to_word, word_to_index = utils.read_vocabulary()

    # Fetch the matrix of word embeddings
    embedding_matrix = utils.read_embedding_matrix(index_to_word)

    # Create the model
    if resume_training:
        model = ChatbotModel(weights_file=weights_file)
    else:
        model = ChatbotModel(embedding_matrix=embedding_matrix)
    
    # Load the data
    q, a = utils.read_training_sequences()

    print("Total training sequences:", q.shape[0])

    print(utils.seq_to_text(q[0], index_to_word))
    print(utils.seq_to_text(a[0], index_to_word))

    q_val = q[:n_val,:]
    a_val = a[:n_val,:]
    q = q[n_val:, :]
    a = a[n_val:, :]
    
    n_train = len(q) - n_val

    step = round(n_train / num_subsets)

    # Prepare validation data
    Q_val, A_val, Y_val = prepare_fit_data(q_val, a_val)

    # Train
    for m in range(EPOCHS):
        print("\nStarting epoch", m, "\n")
        # Loop over training subsets so it fits in RAM
        for n in range(0,n_train,step):
            print('Training epoch: %d, training examples: %d - %d'%(m, n, n + step))
            
            Q,A,Y = prepare_fit_data(q[n:n+step], a[n:n+step])
            model.fit([Q, A], Y, batch_size=BATCH_SIZE, epochs=1)
            
            del Q
            del A
            del Y
            gc.collect()

            """
            for i in range(3):
            test_input = q_test[41+i:42+i]
            print("Context:", utils.seq_to_text(test_input[0], index_to_word))
            print("Answer:", get_response(model, test_input, index_to_word))
            """
        
        print("Evaluating on validation set...")
        loss, acc = model.evaluate([Q_val, A_val], Y_val, verbose=0)
        print("Validation accuracy: %f, loss = %f"%(acc, loss))
                
        model.save_weights(weights_file, overwrite=True)

train()

"""
def batch_generator():
    n = 0
    while True:
        batch_begin = n
        batch_end = min(n + step, round_exem)

        print('training examples: %d - %d'%(batch_begin, batch_end))
            
        Q,A,Y = prepare_fit_data(q[batch_begin:batch_end], a[batch_begin:batch_end])
        n += step

        yield [Q,A],Y

        del Q
        del A
        del Y

model.fit_generator(generator=batch_generator(), steps_per_epoch=1, epochs=EPOCHS)
model.save_weights(weights_file, overwrite=True)
"""
