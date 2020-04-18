import numpy as np
import nltk

import config as cfg
import utils
import textprocessor
from model import ChatbotModel

# Set a random seed for reproducibility
np.random.seed(1337)

index_to_word, word_to_index = utils.read_vocabulary()

# Init our keras model and load the weights from file
weights_file = "model_weights_halftrain_acc62.h5"
model = ChatbotModel(weights_file=weights_file)
print("Model loaded.")

""" Create a sequence from a given raw string. The returned sequence can be fed directly into the bot """
def create_sequence(query):
    # Use NLTK to get word tokens
    tokenized = nltk.word_tokenize(query)
    # Replace out-of-vocabulary words with the UNKNOWN token
    tokenized = [w if w in word_to_index else cfg.TOKEN_UNKNOWN for w in tokenized]
    # Map the words to their respective indices
    X = np.array([word_to_index[w] for w in tokenized])
    # Pad the sequence
    Q = np.zeros((1,cfg.MAX_SEQUENCE_LENGTH))
    if X.size <= cfg.MAX_SEQUENCE_LENGTH:
        Q[0, -X.size:] = X
    else:
        Q[0,:] = X[-cfg.MAX_SEQUENCE_LENGTH:]

    return Q

""" Get the response from the bot, with the given input sequence. """
def get_response_sequence(input_sequence):
    partial_answer = np.zeros((1, cfg.MAX_SEQUENCE_LENGTH))
    partial_answer[0, -1] = cfg.BOS_VOCAB_INDEX
    answer = []
    
    for _ in range(cfg.MAX_SEQUENCE_LENGTH - 1):
        # Predict the next word
        pred_one_hot = model.predict([input_sequence, partial_answer])
        word_index = np.argmax(pred_one_hot)

        # Shift the partial answer by one to the left and set the last word
        partial_answer[0, 0:-1] = partial_answer[0, 1:]
        partial_answer[0, -1] = word_index

        # Add to the answer
        answer.append(word_index)
        if word_index == cfg.EOS_VOCAB_INDEX:  # the index of the symbol EOS (end of sentence)
            break

    # Map the word indices to text
    text = utils.seq_to_text(answer, index_to_word)
    return text


if __name__ == "__main__":
    name = input("Enter your name:")
    print("\n\nCHAT:\n\n")
    print("Bot: hello there, " + name + "!")

    query = ""
    while query != "exit":
        query = input("User: ")
        query = textprocessor.preprocess_query(query, name)
        
        if len(query) > 0:
            Q = create_sequence(query)
            
            # Predict response sequence
            prediction = get_response_sequence(Q)
            start_index = prediction.find("EOS")

            # Format and print
            answer = textprocessor.postprocess_response(prediction[0:start_index], name)
            print("Bot:", answer)