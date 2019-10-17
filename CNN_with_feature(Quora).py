''' Quora dataset
'''
########################################
## import packages
########################################
import os
import re
import csv
import math
import codecs
import numpy
import numpy as np
import pandas as pd
import gensim
import chardet
import keras.backend as K
from string import punctuation
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Merge, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imp import reload
from gensim.models import Word2Vec
from keras.layers import Bidirectional
import metrics
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis
from nltk import word_tokenize
from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from keras.layers.core import Reshape, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.layers.recurrent import GRU

from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler

stop_words = stopwords.words('english')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
import sys
reload(sys)
#sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = 'D:/Code/Quora/data/'
EMBEDDING_FILE = 'D:/Code/Quora/data/quora.bin'
#EMBEDDING_FILE = 'D:/Code/Quora/data/GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

# MAX_SEQUENCE_LENGTH = 20 cho kq cao nhat.
MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 990000
# Su dung quora.bin
EMBEDDING_DIM = 200
# Su dung GoogleNews-vectors-negative300.bin
#EMBEDDING_DIM = 300
batch_size = 128
# epochs 
epochs = 10
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
########################################
## index word vectors
########################################
print('Indexing word vectors')
# Su dung GoogleNews-vectors-negative300.bin
#word2vec = Word2Vec.load_word2vec_format(EMBEDDING_FILE, binary=True)
# Su dung quora.bin
word2vec = Word2Vec.load(EMBEDDING_FILE)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)
def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.    
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        #stops = set(stopwords.words("english"))
        stops.update([',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', ':)', '(:', '``', '...',
                      '-', '``', "''", "'s", "'ve", 'ok', 'hi', 'isnt', "n't", "'m", "'d", 'etc.', '..',
                      '@', 'im', '%', "'ll"])
        text = [lemmatize(w) for w in text if not w in stops and len(w)>3]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, "r", encoding='utf-8', errors='ignore') as f:    
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
labels_test = []
with codecs.open(TEST_DATA_FILE, "r", encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[3]))
        test_texts_2.append(text_to_wordlist(values[4]))
        test_ids.append(values[0])
        labels_test.append(int(values[5]))
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
#print(test_texts_1)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)
labels_test = np.array(labels_test)
print('Shape of data tensor (test):', test_data_1.shape)
print('Shape of label tensor (test):', labels_test.shape)

########################################
## generate leaky features
########################################
train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

ques = pd.concat([train_df[['question1', 'question2']], test_df[['question1', 'question2']]],
                 axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])
def q1_freq(row):
    return(len(q_dict[row['question1']]))    
def q2_freq(row):
    return(len(q_dict[row['question2']]))    
def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)

test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)

########### Moi them doan duoi ##################################
train_df['len_q1'] = train_df.question1.apply(lambda x: len(str(x)))/train_df.question2.apply(lambda x: len(str(x)))
train_df['len_q2'] = train_df.question2.apply(lambda x: len(str(x)))
train_df['diff_len'] = train_df.len_q1 - train_df.len_q2
train_df['len_char_q1'] = train_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_char_q2'] = train_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
train_df['len_word_q1'] = train_df.question1.apply(lambda x: len(str(x).split()))/train_df.question2.apply(lambda x: len(str(x).split()))
train_df['len_word_q2'] = train_df.question2.apply(lambda x: len(str(x).split()))
train_df['common_words'] = train_df.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
train_df['fuzz_qratio'] = train_df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_WRatio'] = train_df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_ratio'] = train_df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
train_df['fuzz_partial_token_sort_ratio'] = train_df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

#### test_df#####
test_df['len_q1'] = test_df.question1.apply(lambda x: len(str(x)))/test_df.question2.apply(lambda x: len(str(x)))
test_df['len_q2'] = test_df.question2.apply(lambda x: len(str(x)))
test_df['diff_len'] = test_df.len_q1 - test_df.len_q2
test_df['len_char_q1'] = test_df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_char_q2'] = test_df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
test_df['len_word_q1'] = test_df.question1.apply(lambda x: len(str(x).split()))/test_df.question2.apply(lambda x: len(str(x).split()))
test_df['len_word_q2'] = test_df.question2.apply(lambda x: len(str(x).split()))
test_df['common_words'] = test_df.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
test_df['fuzz_qratio'] = test_df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_WRatio'] = test_df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_partial_ratio'] = test_df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
test_df['fuzz_partial_token_sort_ratio'] = test_df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

######### Ket thuc doan moi them ############

leaks2 = train_df[['q1_q2_intersect', 'q1_freq', 'q2_freq', 'len_q1', 'diff_len','len_char_q1',
                  'len_char_q2','len_word_q1', 'common_words','fuzz_qratio','fuzz_WRatio',
                  'fuzz_partial_ratio',
                   'fuzz_partial_token_sort_ratio',
                   ]]
test_leaks2 = test_df[['q1_q2_intersect', 'q1_freq', 'q2_freq', 'len_q1', 'diff_len','len_char_q1',
                      'len_char_q2','len_word_q1', 'common_words','fuzz_qratio','fuzz_WRatio',
                      'fuzz_partial_ratio',
                       'fuzz_partial_token_sort_ratio',
                       ]]
ss = StandardScaler()
ss.fit(np.vstack((leaks2, test_leaks2)))
leaks2 = ss.transform(leaks2)
test_leaks2 = ss.transform(test_leaks2)
########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec[word]
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences_1 = embedding_layer(sequence_1_input)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedded_sequences_2 = embedding_layer(sequence_2_input)

# CNN model ################################
conv_layer = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu')

con_1 = conv_layer(embedded_sequences_1)
con_1 = MaxPooling1D(4)(con_1)
con_1 = Flatten()(con_1)
con_1 = Dropout(0.2)(con_1)
con_1 = Dense(50)(con_1)

con_2 = conv_layer(embedded_sequences_2)
con_2 = MaxPooling1D(4)(con_2)
con_2 = Flatten()(con_2)
con_2 = Dropout(0.2)(con_2)
con_2 = Dense(50)(con_2)
#######################
merged1 = concatenate([con_1, con_2])
merged1 = BatchNormalization()(merged1)

merged1 = Dense(200)(merged1)
merged1 = PReLU()(merged1)
merged1 = Dropout(0.3)(merged1)

merged1 = Dense(120)(merged1)
merged1 = PReLU()(merged1)
#merged1 = Dropout(0.2)(merged1)

###########################
leaks2_input = Input(shape=(leaks2.shape[1],))

##feature = concatenate([leaks1_input, leaks2_input])
##feature = BatchNormalization()(feature)
###########################
merged = concatenate([merged1, leaks2_input])
merged = BatchNormalization()(merged)

merged = Dense(133)(merged)
merged = PReLU()(merged)
merged = Dropout(0.2)(merged)

#######
merged = Dense(80)(merged)
merged = PReLU()(merged)
#merged = Dropout(0.2)(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input, leaks2_input], outputs=preds)

sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
# learning schedule callback
# lrate = 0.01
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model.summary()
#print(STAMP)
hist = model.fit([data_1, data_2, leaks2], labels,
                 validation_data=([test_data_1, test_data_2, test_leaks2],labels_test),
                 epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=2)
########################################
## make the submission
########################################
scores = model.evaluate([test_data_1, test_data_2, test_leaks2], labels_test, verbose=0)
print("\nError: %.2f%%" % (100-scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Start making the submission before fine-tuning')
preds = model.predict([test_data_1, test_data_2, test_leaks2], batch_size=batch_size, verbose=0)

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(scores[1]*100)+'_modelCNN_with_feature(Quora).csv', index=False)

# Plot accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#model.save('modelCNN_with_feature(Quora).h5')
