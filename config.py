MAX_SEQUENCE_LENGTH = 30
BOT_NAME = "Rexbot"

# Vocabulary
TOKEN_UNKNOWN = "UNKN"
TOKEN_NONE = "<>"
VOCABULARY_SIZE = 8000
BOS_VOCAB_INDEX = 1
EOS_VOCAB_INDEX = 2

# Model
LSTM_UNITS = 256
WORD_EMBEDDING_SIZE = 50
LEARNING_RATE = 0.001

# File paths

## Dataset
CORNELL_MOVIE_LINES_FILE = "cornell-corpus/movie_lines.txt"
CORNELL_CONVERSATIONS_FILE = "cornell-corpus/movie_conversations.txt"

## Word embeddings
GLOVE_FILE = "glove.6B.50d.txt" 

## Processed data outputs
VOC_FILE = "processed-data/vocabulary.txt"
CONTEXT_FILE = "processed-data/context.txt"
ANSWERS_FILE = "processed-data/answers.txt"
CONTEXT_SEQ_FILE = "processed-data/context_sequences.txt"
ANSWERS_SEQ_FILE = "processed-data/answers_sequences.txt"