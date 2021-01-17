#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.hidden_size = hidden_size
        self.char_embedding_size = char_embedding_size
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size, num_layers=1)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        nn.init.xavier_uniform_(self.char_output_projection.weight)
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, target_vocab.char2id['<pad>'])
        ### END YOUR CODE


    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        x_t = self.decoderCharEmb(input)  # (length, batch_size, embed_size)
        h_t, dec_hidden = self.charDecoder(x_t, dec_hidden)  # (length, batch_size, hidden_size), (1, batch_size, hidden_size) * 2
        scores = self.char_output_projection(h_t)  # (length, batch_size, target_char_size)
        return (scores, dec_hidden)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, _ = self.forward(char_sequence[:-1], dec_hidden)  # (length, batch_size, target_char_size)
        length, batch_size, _ = scores.shape
        loss = F.cross_entropy(scores.reshape((length * batch_size, -1)), char_sequence[1:].reshape((length * batch_size,)),
                               ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        _, batch_size, hidden_size = initialStates[0].shape
        decodeWords = [[self.target_vocab.start_of_word] for _ in range(batch_size)]
        #print(decodeWords)
        decoding_words = decodeWords[:]
        dec_states = initialStates
        word_len = 1
        while decoding_words and word_len <= max_length:
            inputs = torch.tensor([[chars[-1] for chars in decoding_words]], dtype=torch.long, device=device)
            scores, dec_states = self.forward(inputs, dec_states)
            char_indices = scores.squeeze(0).argmax(dim=1)
            #print(char_indices)
            remain_indices = []
            for i, (char_idx, chars) in enumerate(zip(char_indices, decoding_words)):
                if int(char_idx) != self.target_vocab.end_of_word:
                    chars.append(int(char_idx))
                    remain_indices.append(i)
            decoding_words = [decoding_words[i] for i in remain_indices]
            dec_states = (dec_states[0][:, remain_indices, :], dec_states[1][:, remain_indices, :])
            word_len += 1
            #print(decoding_words)
        decodeWords = [''.join([self.target_vocab.id2char[id_] for id_ in chars[1:]])
                       for chars in decodeWords]        
        #print(decodeWords)
        #print(decodeWords)
        return decodeWords
        ### END YOUR CODE

