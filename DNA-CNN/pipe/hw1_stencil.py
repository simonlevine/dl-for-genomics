# HW1 Deep Learning in Genomics
'''
This dataset was synthetically generated for the task of "homotypic motif density localization".
Translation: Finding clusters of the same type of sequence motif

Background:
Transcription Factors (TFs) often bind to particular sequence patterns, known as motifs.
Regulatory regions of DNA often have more than one binding site/motif for a particular TF clustered in a small region.
The small dataset provided has sequences classified as either positive or negative depending on where there are clusters of a certain motif.

Task:
1) Train a CNN to classify the sequences as positive or negative for homotypic motif clusters
    Your architecture must have 1 Convolution layer (+ nonlinearity + pooling) and 1 dense output layer
        Only a small number of hidden units are needed for this task
    Use the provided function for obtaining nucleotide one-hot encodings of the sequences
    Implement the provided functions train_cnn() and evaluate_model() to train and test the model

2) Plot the training and validation loss after each epoch of training.
    
3) Implement a 2-layer fully connected neural network architecture (with non-linearity) to perform binary classification.
    Implement the provided function kmer_counts() to preprocess the data
    Implement the provided function train_kmer_counts_nn() to train the model
    Use evaluate_model() to test the model

Bonus:
Implement a Convolutional Neural Network (your choice of number/size of layers) for binary classification that uses an embedding layer to encode a DNA sequence.
You should create helper functions to preprocess the data and train the model, as were used in the previous parts.
For implementation of the embeddings, use embedding layers in your framework of choice.
Pytorch: https://pytorch.org/docs/stable/nn.html#embedding
Keras: https://keras.io/layers/embeddings/
Tensorflow: https://www.tensorflow.org/tutorials/text/word_embeddings
'''

import numpy as np

# Example imports of Pytorch & TF/Keras frameworks:
# import torch
# import torch.nn as nn
# import tensorflow as tf

def get_data(filename='hw1_data.npz'):
    all_data = np.load(filename)
    train_seq = all_data['train_seq']
    train_y = all_data['train_y']
    valid_seq = all_data['valid_seq']
    valid_y = all_data['valid_y']
    test_seq = all_data['test_seq']
    test_y = all_data['test_y']
    
    return train_seq, train_y, valid_seq, valid_y, test_seq, test_y


def one_hot_encoding(seq_array: np.ndarray) -> np.ndarray:
    """
    :param seq_array: np array of DNA sequences
    :return: np array of one-hot encodings of input DNA sequences
    """
    nuc2id = {'A' : 0, 'C' : 1, 'T' : 2, 'G' : 3}
    onehot_array = np.zeros((len(seq_array), 4, 1500))
    for seq_num, seq in enumerate(seq_array):
        for seq_idx, nucleotide in enumerate(seq):
            nuc_idx = nuc2id[nucleotide]
            onehot_array[seq_num, nuc_idx, seq_idx] = 1
    
    return onehot_array


def kmer_counts(seq_array: np.ndarray, k: int) -> np.ndarray:
    """
    :param seq_array: np array of DNA sequences
    :param k: length of k-mers
    :return: np array of k-mer counts per sequence
    """
    # TODO
    pass

def train_cnn(train_onehot_array, train_y, valid_onehot_array, valid_y):
    """
    Implements and trains a CNN with 1 convolution layer (including non-linearity and pooling)
    followed by 1 dense output layer
    param train_onehot_array: np array of one-hot encodings of input DNA sequences for training
    param train_y: np array of training labels
    param valid_onehot_array: np array of one-hot encodings of input DNA sequences for validation
    param valid_y: np array of validation labels
    return: the trained model
    """
    
    # TODO: Define the model

    # Example with keras:
    # model = tf.keras.Sequential()
    # model.add(keras_layer)
    
    
    
    # TODO: Train the model
    
    # Example with keras:
    # model.compile(arguments)
    # model.fit(arguments)
    
    
    
    # TODO: Return the trained model
    
    # Example with keras:
    # return model
    
    pass

def train_kmer_counts_nn(train_counts, train_y, valid_counts, valid_y):
    """
    param train_counts: np array of kmer counts in input DNA sequences for training
    param train_y: np array of training labels
    param valid_counts: np array of kmer counts in input DNA sequences for validation
    param valid_y: np array of validation labels
    return: a trained model using kmer count input data
    """
    # TODO: Define the model
    
    # TODO: Train the model
    
    # TODO: Return the trained model
    
    pass
                

def evaluate_model(trained_model, test_inputs, test_y):
    """
    Prints the % accuracy of the model on the test data
    param trained_model: a trained model
    param test_inputs: np array of model inputs in the test set
    param test_y: np array of test labels
    """
    
    # TODO
    
    # Example with keras:
    # model.evaluate(arguments)
    
    pass

def main():
    
    # TODO: Call get_data() to read in all of the data
    
    # TODO: For each neural network:
    # 1) Preprocess the input sequences into the correct model inputs using the appropriate helper function
    # 2) Train the model using the appropriate train function
    # 3) Evaluate the model's performance using evaluate_model()
    
    pass
    
if __name__ == '__main__':
    main()
        