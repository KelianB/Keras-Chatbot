# Keras-Chatbot
A chatbot written in Python using Tensorflow / Keras.

## Running the chatbot

### Requirements
The bot was written for python 3.6.9, using tensorflow v2.2 and NLTK v3.2.5.

## Training
In order to train the bot yourself, you need to get the GloVe word embeddings [here](https://nlp.stanford.edu/projects/glove/)(download the glove.6B.zip archive and add glove.6B.50d.txt to the root of this project). 
You can then run train.py to train the bot. 
You might first need to tweak some of the training parameters in train.py so it fits on your hardware (most importantly `num_subsets`, which splits the training data into smaller sets so it can fit in RAM).

The model was originally trained on Google Colab on a Tesla P100-PCIE-16GB (coreClock: 1.3285GHz, coreCount: 56, deviceMemorySize: 15.90GiB, deviceMemoryBandwidth: 681.88GiB/s)

### Re-generating the training data
If you wish to make changes to the maximum sequence size, vocabulary or formatting of the dataset, you will have to re-generate the training data. You can download the cornell movie-dialogs corpus [here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and include the movie_conversations.txt and movie_lines.txt files in the cornell-corpus folder. You should then be able to run create-training-data.py to re-generate the training data and vocabulary.


