import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional,  Concatenate
from keras.utils import *
from keras.initializers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import tensorflow as tf
import sys
import time
import csv

t0 = time.time()

PAD_token = 0   # Used for padding short sentences
SOS_token = 1   # Start-of-sentence token
EOS_token = 2   # End-of-sentence token
OOV_Token = 3
top_k = 1000
sentence_length = 68
batch_size = 64
latent_dim = 256
epochs = 100
embedding_size = 100
next_index1 = 4
next_index2 = 4

class Vocabulary:

    def __init__(self, name):

        self.name = name
        self.word2index = {"PAD":PAD_token, "SOS": SOS_token, "EOS": EOS_token, "OOV": OOV_Token}
        self.word2count = {}
        self.index2word = { PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS",OOV_Token:"OOV" }
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1
            
col_list1 = ["sourceLineTokens","targetLineTokens"]
df1 = pd.read_csv('C:\\Users\\HP\\Assignment 2\\train.csv',usecols=col_list1)
num_samples = len(df1.index)

voc1 = Vocabulary('name')

for i in range(num_samples):
    voc1.add_sentence(eval(df1[col_list1[0]].iloc[i]))
for i in range(num_samples):
    voc1.add_sentence(eval(df1[col_list1[1]].iloc[i]))
    
next_index1 = 4
sorted_age = sorted(voc1.word2count.items(), key = lambda kv: kv[1],reverse=True)
dictionary_of_tokens1 = { "PAD" : PAD_token, "SOS": SOS_token, "EOS": EOS_token, "OOV": OOV_Token}
for token in sorted_age:
    if(next_index1<=top_k+3):
        dictionary_of_tokens1[token[0]] = next_index1
        next_index1+=1
    else:
         dictionary_of_tokens1[token[0]] = OOV_Token


def listcon(df1,x,number):
    list1 = []
    for i in range(number):
        l = eval(df1[col_list1[x]].iloc[i])
        l = ['SOS'] + l + ['EOS']
        list1.append(l)
    return list1
list_source = listcon(df1,0,num_samples)
list_target = listcon(df1,1,num_samples)

def tokanize(list1,number):
    token_list1 =[]
    for i in range(number):
        l = list1[i]
        ll = []
        for token in l:
            try:
                x = dictionary_of_tokens1[token]
            except KeyError:
                x = OOV_Token
            ll.append(x)
        token_list1.append(ll)
    return token_list1
token_list_source = tokanize(list_source,num_samples)
token_list_target = tokanize(list_target,num_samples)

padded_target_token = tf.keras.preprocessing.sequence.pad_sequences(token_list_target,padding='post', maxlen = sentence_length)
padded_source_token = tf.keras.preprocessing.sequence.pad_sequences(token_list_source,padding='post', maxlen = sentence_length)

top_k = next_index1
vocab_size = len(dictionary_of_tokens1)

encoder_input_data = np.zeros((len(padded_source_token), sentence_length),dtype='float32')
decoder_input_data = np.zeros((len(padded_source_token), sentence_length), dtype='float32')
decoder_target_data = np.zeros((len(padded_source_token), sentence_length, top_k),dtype='float32')


for i, (input_sent, target_sent) in enumerate(zip(padded_source_token, padded_target_token)):
    for t in range(len(input_sent)):
        encoder_input_data[i, t] = input_sent[t]
    for t in range(len(target_sent)):
        decoder_input_data[i, t] = target_sent[t]
        if t > 0:
            decoder_target_data[i, t - 1, target_sent[t]] = 1.
            
model2 = tf.keras.models.load_model('C:\\Users\\HP\\Desktop\\ASEMLDATA')

#Inference Encoder
encoder_inputs = model2.input[0]
output_encoder, state_h, state_c = model2.layers[4].output
encoder_states = [state_h, state_c]
encoder_model2 = Model(encoder_inputs, encoder_states)

#Inference Decoder
decoder_inputs = model2.input[1]
decoder_state_input_h = Input(shape=(latent_dim,),name = "name1")
decoder_state_input_c = Input(shape=(latent_dim,),name = "name2")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding1 = model2.layers[3]
final_decoder_embedding = decoder_embedding1(decoder_inputs)
decoder_lstm1 = model2.layers[5]
decoder_outputs2, state_h2, state_c2 = decoder_lstm1(final_decoder_embedding, initial_state= decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_dense2 = model2.layers[6]
decoder_outputs2 = decoder_dense2(decoder_outputs2)
decoder_model2 = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

rev_dictionary_of_tokens1 = dict([ (i,char) for char,i in dictionary_of_tokens1.items()])
#print(dictionary_of_tokens1)
#print(rev_dictionary_of_tokens1)
rev_dictionary_of_tokens1[3]='OOV_Token'
#print(rev_dictionary_of_tokens1)
def decode_sequence(input_seq):

    states_value = encoder_model2.predict(input_seq)
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = dictionary_of_tokens1['SOS']

    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        #print(2)
        output_tokens, h, c = decoder_model2.predict(
            [target_seq] + states_value)
        #print(output_tokens.shape)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #print(sampled_token_index)
        if(sampled_token_index==OOV_Token):
            sampled_char = 'OOV_Token'
        sampled_char = rev_dictionary_of_tokens1[sampled_token_index]
        if (sampled_char == 'EOS' or len(decoded_sentence) > maxSentenceLen): 
                stop_condition = True
                break
        decoded_sentence.append(sampled_char)
        

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


filename = sys.argv[0]
#sys.argv[0] "demo_valid.py" 
pathname_source = sys.argv[1]
#sys.argv[1] "valid.csv" 
pathname_prediction = sys.argv[2]
 #sys.argv[2] "output_file.csv"
print(filename,pathname_source, pathname_prediction)
col_list1 = ["sourceLineTokens"]
df2 = pd.read_csv(pathname_source, usecols=col_list1)
num_samples_valid = len(df2.index)
print('success')
list_source_valid = listcon(df2,0,num_samples_valid)
token_list_source_valid = tokanize(list_source_valid,num_samples_valid)
padded_source_token_token = tf.keras.preprocessing.sequence.pad_sequences(token_list_source_valid,padding='post', maxlen = sentence_length)
encoder_input_data_valid = np.zeros((len(padded_source_token_token), sentence_length),dtype='float32')
for i, input_sent in enumerate(padded_source_token_token):
    for t in range(len(input_sent)):
        encoder_input_data_valid[i, t] = input_sent[t]


t0= time.time()
with open(pathname_prediction, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sourceLineTokens", "fixedTokens"])
    for seq_index in range(num_samples_valid):
        input_seq = encoder_input_data_valid[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        writer.writerow([df2[col_list1[0]].iloc[seq_index],  decoded_sentence ])

t1 = time.time() - t0
print("Time elapsed: ", t1)
print('success')  
