import tensorflow as tf
import tensorflow.keras.models

print("TF Version: {}".format(tf.version.VERSION))
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import  Tokenizer
from tensorflow.keras.preprocessing.sequence import  pad_sequences
import numpy as np
from tensorflow.keras.layers import Embedding,Dense,LSTM,GRU,Bidirectional,Flatten,GlobalAveragePooling1D,Conv2D

imdb, info = tfds.load("imdb_reviews",as_supervised=True,with_info=True)


train_data,test_data=imdb['train'],imdb['test']



print(type(train_data))

train_sentence=[]
train_labels=[]

test_sentence=[]
test_labels=[]


for s,l in train_data:
    train_sentence.append(str(s.numpy()))
    train_labels.append((l.numpy()))

for s,l in test_data:
    test_sentence.append(str(s.numpy()))
    test_labels.append((l.numpy()))

print("Train Sentence: {}, Train Labels: {}".format(len(train_sentence),len(train_labels)))
print("Test Sentence: {}, Test Labels: {}".format(len(test_sentence),len(test_labels)))

train_labels= np.array(train_labels)
test_labels= np.array(test_labels)

vocab_size=10000
oov_word="<OOV>"
max_len=50
embed_dims=16

token_obj = Tokenizer(num_words=vocab_size,oov_token=oov_word)
token_obj.fit_on_texts(train_sentence)

pad_train_seq = pad_sequences(token_obj.texts_to_sequences(train_sentence),maxlen=max_len,padding="post",
                              truncating="post")

pad_test_seq = pad_sequences(token_obj.texts_to_sequences(test_sentence),maxlen=max_len,padding="post",
                              truncating="post")

model = tensorflow.keras.models.Sequential([
    Embedding(vocab_size,embed_dims,input_length=max_len),
    Flatten(),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(pad_train_seq,train_labels, epochs=10,validation_data=(pad_test_seq,test_labels))

print('Accuracy Flatten: {}'.format(model.evaluate(pad_test_seq,test_labels,verbose=0)[1]*100))
import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,2,figsize=(12,5))
epochs=range(len(history.history['accuracy']))

ax[0].plot(epochs,history.history['accuracy'],label='train',lw=2,marker='o',color='tomato')
ax[0].plot(epochs,history.history['val_accuracy'],label='val',lw=2,marker='*',color='teal')
ax[0].set_title('Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()


ax[1].plot(epochs,history.history['loss'],label='train',lw=2,marker='o',color='tomato')
ax[1].plot(epochs,history.history['val_loss'],label='val',lw=2,marker='o',color='teal')
ax[1].set_title('Loss')
ax[1].legend()

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
plt.suptitle('Learning Curves-Flatten')
plt.show()


## Model-2

model = tensorflow.keras.models.Sequential([
    Embedding(vocab_size,embed_dims,input_length=max_len),
    #Flatten(),
    GlobalAveragePooling1D(),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(pad_train_seq,train_labels, epochs=10,validation_data=(pad_test_seq,test_labels))
print('Accuracy Flatten+Gloabl: {}'.format(model.evaluate(pad_test_seq,test_labels,verbose=0)[1]*100))

import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,2,figsize=(12,5))
epochs=range(len(history.history['accuracy']))

ax[0].plot(epochs,history.history['accuracy'],label='train',lw=2,marker='o',color='tomato')
ax[0].plot(epochs,history.history['val_accuracy'],label='val',lw=2,marker='*',color='teal')
ax[0].set_title('Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()


ax[1].plot(epochs,history.history['loss'],label='train',lw=2,marker='o',color='tomato')
ax[1].plot(epochs,history.history['val_loss'],label='val',lw=2,marker='o',color='teal')
ax[1].set_title('Loss')
ax[1].legend()

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
plt.suptitle('Learning Curves-Flatten_GloablAvg')
plt.show()


## Model-3

model = tensorflow.keras.models.Sequential([
    Embedding(vocab_size,embed_dims,input_length=max_len),
    Bidirectional(LSTM(32)),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(pad_train_seq,train_labels, epochs=10,validation_data=(pad_test_seq,test_labels))
print('Accuracy LSTM: {}'.format(model.evaluate(pad_test_seq,test_labels,verbose=0)[1]*100))

import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,2,figsize=(12,5))
epochs=range(len(history.history['accuracy']))

ax[0].plot(epochs,history.history['accuracy'],label='train',lw=2,marker='o',color='tomato')
ax[0].plot(epochs,history.history['val_accuracy'],label='val',lw=2,marker='*',color='teal')
ax[0].set_title('Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()


ax[1].plot(epochs,history.history['loss'],label='train',lw=2,marker='o',color='tomato')
ax[1].plot(epochs,history.history['val_loss'],label='val',lw=2,marker='o',color='teal')
ax[1].set_title('Loss')
ax[1].legend()

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
plt.suptitle('Learning Curves-LSTM')
plt.show()


## Model-3

model = tensorflow.keras.models.Sequential([
    Embedding(vocab_size,embed_dims,input_length=max_len),
    Bidirectional(GRU(32)),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(pad_train_seq,train_labels, epochs=10,validation_data=(pad_test_seq,test_labels))
print('Accuracy GRU: {}'.format(model.evaluate(pad_test_seq,test_labels,verbose=0)[1]*100))

import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,2,figsize=(12,5))
epochs=range(len(history.history['accuracy']))

ax[0].plot(epochs,history.history['accuracy'],label='train',lw=2,marker='o',color='tomato')
ax[0].plot(epochs,history.history['val_accuracy'],label='val',lw=2,marker='*',color='teal')
ax[0].set_title('Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()


ax[1].plot(epochs,history.history['loss'],label='train',lw=2,marker='o',color='tomato')
ax[1].plot(epochs,history.history['val_loss'],label='val',lw=2,marker='o',color='teal')
ax[1].set_title('Loss')
ax[1].legend()

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
plt.suptitle('Learning Curves-GRU')
plt.show()


## Model-4

model = tensorflow.keras.models.Sequential([
    Embedding(vocab_size,embed_dims,input_length=max_len),
    Conv2D(128,5,activation='relu'),
    GlobalAveragePooling1D(),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(pad_train_seq,train_labels, epochs=10,validation_data=(pad_test_seq,test_labels))
print('Accuracy Flatten: {}'.format(model.evaluate(pad_test_seq,test_labels,verbose=0)[1]*100))

import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,2,figsize=(12,5))
epochs=range(len(history.history['accuracy']))

ax[0].plot(epochs,history.history['accuracy'],label='train',lw=2,marker='o',color='tomato')
ax[0].plot(epochs,history.history['val_accuracy'],label='val',lw=2,marker='*',color='teal')
ax[0].set_title('Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()


ax[1].plot(epochs,history.history['loss'],label='train',lw=2,marker='o',color='tomato')
ax[1].plot(epochs,history.history['val_loss'],label='val',lw=2,marker='o',color='teal')
ax[1].set_title('Loss')
ax[1].legend()

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
plt.suptitle('Learning Curves-CONV')
plt.show()