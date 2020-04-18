import numpy as np

import config as cfg

""" Read the vocabulary from file, as a tuple of index_to_word array, word_to_index dict. """
def read_vocabulary():
    with open(cfg.VOC_FILE) as voc_file:
        index_to_word = [line.strip() for line in voc_file]
        voc_file.close()

        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        return index_to_word, word_to_index


""" Read the training sequences from file, as a tuple of np arrays context_sequences, answer_sequences. """
def read_training_sequences():
    # Read context
    with open(cfg.CONTEXT_SEQ_FILE) as context_file:
        context_sequences = [[int(i) for i in line.split(",")] for line in context_file]
        context_file.close()

    # Read answers
    with open(cfg.ANSWERS_SEQ_FILE) as answers_file:
        answers_sequences = [[int(i) for i in line.split(",")] for line in answers_file]
        answers_file.close()

    return np.array(context_sequences), np.array(answers_sequences)


""" Read the word embedding matrix from GloVe. """
def read_embedding_matrix(index_to_word):
    embeddings_index = {}

    # Read file
    with open(cfg.GLOVE_FILE, encoding="utf8") as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        glove_file.close()

    print("Loaded embedding with %s word vectors." % len(embeddings_index))
    
    # Create matrix
    embedding_matrix = np.zeros((cfg.VOCABULARY_SIZE, cfg.WORD_EMBEDDING_SIZE))
    words_without_embedding = 0

    for i, word in enumerate(index_to_word):
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index.get(word)
        else:
            # words not found in embedding index are assigned a vector of zeros
            words_without_embedding += 1
    
    print(words_without_embedding, "out of", cfg.VOCABULARY_SIZE, "words in the vocabulary do not have a default embedding vector.")
        
    del embeddings_index

    return embedding_matrix


""" Converts a sequence of word indices to text. """
def seq_to_text(indices, index_to_word):
    text = ""
    for k in indices:
        text += (index_to_word[k] if k < len(index_to_word) else str(k)) + " "
    return text