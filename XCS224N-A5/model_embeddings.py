#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.vocab = vocab
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), embed_size, pad_token_idx)
        self.conv = CNN(embed_size, 5)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        output = []
        for x_padded in input_tensor.split(1, dim=0):
            x_padded = x_padded.squeeze(0)  # (batch_size, max_word_length)
            x_emb = self.embeddings(x_padded)  # (batch_size, max_word_length, embed_size)
            x_reshaped = x_emb.transpose(1, 2)  # (batch_size, embed_size, max_word_length)
            x_convout = self.conv(x_reshaped)  # (batch_size, embed_size)
            x_highway = self.highway(x_convout)  # (batch_size, embed_size)
            x_word_emb = self.dropout(x_highway)  # (batch_size, embed_size)
            output.append(x_word_emb)
        output = torch.stack(output)
        return output
        ### END YOUR CODE
