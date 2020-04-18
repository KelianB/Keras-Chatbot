import numpy as np
import gc

import config as cfg
import utils
from model import ChatbotModel

# Training parameters
EPOCHS = 30
BATCH_SIZE = 512
NUM_SUBSETS = 10
N_VAL = 2000

weights_file = "model_weights.h5"

# If true, the model will be initialized with the weights from the line above.
resume_training = False


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
    index_to_word, _ = utils.read_vocabulary()

    # Create the model
    if resume_training:
        model = ChatbotModel(weights_file=weights_file)
    else:
        model = ChatbotModel(embedding_matrix=utils.read_embedding_matrix(index_to_word))
    
    # Load the data
    q, a = utils.read_training_sequences()

    print("Total training sequences:", q.shape[0])

    print("Example context-answer pair")
    print(utils.seq_to_text(q[0], index_to_word))
    print(utils.seq_to_text(a[0], index_to_word))

    q_val = q[:N_VAL,:]
    a_val = a[:N_VAL,:]
    q = q[N_VAL:, :]
    a = a[N_VAL:, :]
    
    n_train = len(q) - N_VAL

    step = round(n_train / NUM_SUBSETS)

    # Prepare validation data
    Q_val, A_val, Y_val = prepare_fit_data(q_val, a_val)

    # Train
    for m in range(EPOCHS):
        print("\nStarting epoch", m+1, "\n")
        # Loop over training subsets so it fits in RAM
        for n in range(0,n_train,step):
            print("Training epoch: %d. Data slice: %d - %d" % (m+1, n, n + step))
            
            Q,A,Y = prepare_fit_data(q[n:n+step], a[n:n+step])
            model.fit([Q, A], Y, batch_size=BATCH_SIZE, epochs=1)

            # Make sure memory is cleared 
            del Q
            del A
            del Y
            gc.collect()
        
        print("Evaluating on validation set...")
        loss, acc = model.evaluate([Q_val, A_val], Y_val, verbose=0)
        print("Validation accuracy: %f, loss = %f"%(acc, loss))
                
        model.save_weights(weights_file, overwrite=True)

if __name__ == "__main__":
    train()