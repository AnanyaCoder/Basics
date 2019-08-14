#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:32:46 2019

@author: ananya
"""
import numpy as np

def word_vec(word,word_index,w1):
    w_index = word_index[word]
    v_w = w1[w_index]
    return v_w

def word2onehot(word,unique_words,word_index):
    # word_vec - initialise a blank vector
    word_vec = np.zeros(unique_words) 
    # Get ID of word from word_index
    word_index = word_index[word]
    # Change value from 0 to 1 according to ID of the word
    word_vec[word_index] = 1
    return word_vec

def GenerateTrainingData(tokens,settings):
    #finding unique word counts using dictionary
    word_cnts = {}
    for row in tokens:
        for word in row:
            if word not in word_cnts:
                word_cnts[word] = 1
            else:
                word_cnts[word] += 1
    
    unique_words = len(word_cnts.keys())
    #Generate lookup dictionaries
    word_list = list(word_cnts.keys())
    # Generate word:index
    word_index = dict((word, i) for i, word in enumerate(word_list))
    # Generate index:word
    index_word = dict((i, word) for i, word in enumerate(word_list))
    
    training_data = []
    # Cycle through each sentence in corpus
    for sentence in tokens:
        
        sent_len = len(sentence)
        # Cycle through each word in sentence
        for i,word in enumerate(sentence):
            # Convert target word to one-hot
            w_target = word2onehot(sentence[i],unique_words,word_index)
            # Cycle through context window
            w_context = []
                       
            for j in range(i-settings['window_size'],i+settings['window_size']+1):
              # Criteria for context word 
              # 1. Target word cannot be context word (j != i)
              # 2. Index must be greater or equal than 0 (j >= 0) 
              #                       - if not list index out of range
              # 3. Index must be less or equal than length of sentence 
              #                      (j <= sent_len-1) - if not list index out of range 
          
                if j != i and j <= sent_len-1 and j >= 0:
                   # Append the one-hot representation of word to w_context
                   w_context.append(word2onehot(sentence[j],unique_words,word_index))
                   print(sentence[j],sentence[i],) 
                   training_data.append([w_target, w_context])
    return np.array(training_data),unique_words,word_index,index_word

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def forward_pass(ContextVec,w1,w2):
    h = np.dot(ContextVec.T,w1)   #hidden layer
    u = np.dot(h.T, w2)           #output layer
    return softmax(u),h,u

def backward_pass(error,h,w_t,lr,w1,w2):
    #h - shape 10x1, e - shape 7x1, dl_dw2 - shape 10x7
    dl_dw2 = np.outer(h, error.T)
    dl_dw1 = np.outer(w_t, np.dot(w2, error).T)
    w1 = w1 - (lr * dl_dw1)
    w2 = w2 - (lr * dl_dw2)
    return w1,w2

def vec_sim(vec,unique_words,index_word):
    v_w1 = vec
    word_sim = {}

    for i in range(unique_words):
      # Find the similary score for each word in vocab
      v_w2 = w1[i]
      theta_sum = np.dot(v_w1, v_w2)
      theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
      theta = theta_sum / theta_den

      word = index_word[i]
      word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

    for word, sim in words_sorted:
      print(word, sim)

#text = "I love bread and butter and pasta and sandwiches"
text = "I love to go to school and attend my classes"
#text = "The quick brown fox jumped over the lazy dogs"
#text = "Natural Language processing and machine learning is fun and exciting"
tokens = [[word.lower() for word in text.split()]]

settings = {
	'window_size': 2,	# context window +- center word (2 to the left & 2 to the right of target word)
	'n': 10,		# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 100,		# number of training epochs
	'learning_rate': 0.01	# learning rate
}

training_data,unique_words,word_index,index_word =  GenerateTrainingData(tokens,settings)

w1 = np.random.uniform(-1, 1, (unique_words, settings['n']))
w2 = np.random.uniform(-1, 1, (settings['n'] ,unique_words))
for i in range(settings['epochs']):
    loss = 0 #initialize loss to zero
    for w_t, w_c in training_data: #target word vector, context word vector
        x = np.mean(w_c,axis=0) #mean of all the context words
        y_pred, hiddenLayer, ouputLayer = forward_pass(x,w1,w2)
        EI = np.subtract(y_pred, w_t) 
        w1,w2 = backward_pass(EI, hiddenLayer, x,settings['learning_rate'],w1,w2)
                
        loss += -np.sum([ouputLayer[np.where(w_t == 1)[0][0]]]) + len(w_t) * np.log(np.sum(np.exp(ouputLayer)))
    print('Epoch:', i, "Loss:", loss)

text_context = ['to','school','attend','my']
#text_context = ['jumped','over','brown']
test_vec = []
for word in text_context:
    test_vec.append(word_vec(word,word_index,w1))

final_test_vec = np.mean(test_vec,axis = 0)

vec_sim(final_test_vec,unique_words,index_word)
    