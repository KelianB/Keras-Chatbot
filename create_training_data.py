import re
import numpy as np
from collections import Counter
import nltk

import config as cfg
import textprocessor

PROP_SEPARATOR = "+++$+++"


""" Create a dictionary of movie_line_id:movie_line_text entries. """ 
def create_movie_lines_dict():
    lines_file = open(cfg.CORNELL_MOVIE_LINES_FILE, encoding="iso-8859-1")

    lines = dict()

    for l in lines_file:
        # Example input: L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
        split = l.split(PROP_SEPARATOR)
        line_id = split[0].strip()
        line_string = split[-1].strip()
        
        # Add entry to the dict
        lines[line_id] = line_string
        
    lines_file.close()
    return lines


""" 
    Create the vocabulary from the movie lines.
    Return a tuple: index_to_word (list), word_to_index (dict).
"""
def create_vocab(output_file_name, lines, vocab_size):
    extra_tokens = [cfg.TOKEN_NONE, "BOS", "EOS", cfg.TOKEN_UNKNOWN]
    
    # Compute most common words
    vocab = Counter()
    print("Computing most common words in dataset...")
    for text in lines:
        words = nltk.word_tokenize(text)
        vocab.update(w.lower() for w in words)

    most_common = vocab.most_common(vocab_size - len(extra_tokens))

    # Create index_to_word array: most common words in corpus and some extra tokens
    index_to_word = extra_tokens + [item[0] for item in most_common]
    
    # Create word_to_index dict
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print("The least frequent word in the vocabulary is '%s', with %d occurrences. (vocab_size = %d)" % (index_to_word[-1], most_common[-1][1], vocab_size))

    # Output vocab to file
    output_file = open(output_file_name, "w", encoding="utf8")
    for i in range(len(index_to_word)):
        output_file.write(("\n" if i > 0 else "") + index_to_word[i])
    output_file.close()

    return index_to_word, word_to_index


""" 
    Get a list of conversations from the dataset.
    Each conversation is a list of movie lines in order.
"""
def read_conversations(idx_to_movie_line):
    conversation_file = open(cfg.CORNELL_CONVERSATIONS_FILE, encoding="utf8")
    conversations = []

    for l in conversation_file:
        # Extract the IDs of the lines of that conversation 
        # For example: "u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']" will yield [L194, L195, L196, L197]
        line_ids = re.search(r"\[(.*?)\]", l).group(1).replace("'", "").split(",")
        line_ids = [line_id.strip() for line_id in line_ids]
        conversations.append([idx_to_movie_line[line_id] for line_id in line_ids])
    
    conversation_file.close()
    return conversations


"""
    Create a list of cleaned-up context and answers and output them to their respective files.
    Returns tuple: context (list), answers (list)
"""
def create_context_answers(conversations, context_output_file, answers_output_file):
    context = []
    answers = []

    for conv in conversations:
        # First, pre-process the lines
        for j in range(len(conv)):
            conv[j] = textprocessor.cleanup(conv[j])

            # Fix lines endings that only work in spoken dialogs
            delete_endings = [" and", " or"]
            for e in delete_endings:
                l = len(e)
                if len(conv[j]) > l and conv[j][-l:] == e:
                    conv[j] = conv[j][:-l]
  
            # Clean-up again (in case we have created some inconsistencies with the previous actions)
            conv[j] = textprocessor.cleanup(conv[j])

    for conv in conversations:
        for i in range(1, len(conv)):
            if len(conv[i]) > 0 and len(conv[i-1]) > 0:
                context.append(conv[i-1])
                answers.append(conv[i])
    
    # Output to files
    con_file = open(context_output_file, "w", encoding="utf8")
    ans_file = open(answers_output_file, "w", encoding="utf8")
    
    for i in range(len(context)):
        con_file.write(("\n" if i > 0 else "") + context[i])
    for i in range(len(answers)):
        ans_file.write(("\n" if i > 0 else "") + answers[i])

    con_file.close()
    ans_file.close()
    
    return context, answers


""" Convert the given answers and context into sequences of integers, and output to respective files. """
def create_context_answers_sequences(context, answers, word_to_index, context_seq_output_file, answers_seq_output_file):
    answers = ["BOS " + sent + " EOS" for sent in answers]

    print("Tokenizing context and answers...")
    tokenized_context = [nltk.word_tokenize(sent) for sent in context]
    tokenized_answers = [nltk.word_tokenize(sent) for sent in answers]
    
    """   
    # Replace all words that are not in the vocabulary with the unknown token:    
    for i, sent in enumerate(tokenized_context):
        tokenized_context[i] = [w if w in word_to_index else TOKEN_UNKNOWN for w in sent]

    for i, sent in enumerate(tokenized_answers):
        tokenized_answers[i] = [w if w in word_to_index else TOKEN_UNKNOWN for w in sent]
    """

    def filter_tokens(items_1, items_2, filter_function, print_str):
        filtered_1 = []
        filtered_2 = []
        for i in range(len(items_1)):
            x1, x2 = items_1[i], items_2[i]
            if (not filter_function(x1)) and (not filter_function(x2)):
                filtered_1.append(x1)
                filtered_2.append(x2)

        num_filtered = len(items_1) - len(filtered_1)
        assert num_filtered == (len(items_2) - len(filtered_2))
        percent_filtered = 100 * num_filtered / (len(tokenized_context)+1e-5)
        print(num_filtered, "out of", len(items_1), "(" + str(percent_filtered) + "%)", print_str) 
        
        return filtered_1, filtered_2
    
    tokenized_context, tokenized_answers = filter_tokens(tokenized_context, tokenized_answers, 
        lambda x: len(x) > cfg.MAX_SEQUENCE_LENGTH,
        "entries with either context or answer sentences larger than max sequence length."
    )
    
    def is_outside_vocab(tokens):
        for t in tokens:
            if not(t in word_to_index):
                return True
        return False

    tokenized_context, tokenized_answers = filter_tokens(tokenized_context, tokenized_answers, 
        is_outside_vocab, "entries with either context or answer sentences out of vocabulary."
    )

    # Creating the training data:
    X = np.array([[word_to_index[w] for w in sent] for sent in tokenized_context])
    Y = np.array([[word_to_index[w] for w in sent] for sent in tokenized_answers])

    from tensorflow.keras.preprocessing import sequence
    Q = sequence.pad_sequences(X, maxlen=cfg.MAX_SEQUENCE_LENGTH)
    A = sequence.pad_sequences(Y, maxlen=cfg.MAX_SEQUENCE_LENGTH, padding="post")

    # Output to files
    con_file = open(context_seq_output_file, "w", encoding="utf8")
    ans_file = open(answers_seq_output_file, "w", encoding="utf8")

    for i in range(len(Q)):
        con_file.write(("\n" if i > 0 else "") + ",".join(Q[i].astype("str")))
    for i in range(len(A)):
        ans_file.write(("\n" if i > 0 else "") + ",".join(A[i].astype("str")))

    con_file.close()
    ans_file.close()


if __name__ == "__main__":
    print("Creating dict of movie lines...")
    movie_lines_dict = create_movie_lines_dict()
    print("Reading conversations file...")
    conversations = read_conversations(movie_lines_dict)
    print("Creating context and answers...")
    context, answers = create_context_answers(conversations, cfg.CONTEXT_FILE, cfg.ANSWERS_FILE)
    print("Creating vocabulary...")
    index_to_word, word_to_index = create_vocab(cfg.VOC_FILE, context + answers, cfg.VOCABULARY_SIZE)

    create_context_answers_sequences(context, answers, word_to_index, cfg.CONTEXT_SEQ_FILE, cfg.ANSWERS_SEQ_FILE)
    print("Done!")