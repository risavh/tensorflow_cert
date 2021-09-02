import tensorflow as tf
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import  pad_sequences


print('TF Version: {}'.format(tf.version.VERSION))
print('GPU','avaliable' if tf.config.list_physical_devices('GPU') else 'not avaliable')

path=r"D:\works\koglu\DL_Training\sarcasm\archive (1)\Sarcasm_Headlines_Dataset_v2.json"

f = open(path,'r')
data=f.readlines()
f.close()
print(len(data),data[:3])

txt_lst=[]
lbl_lst=[]
for i in data:
    row_wise = json.loads(i)
    lbl_lst.append(row_wise['is_sarcastic'])
    txt_lst.append(row_wise['headline'])

print('# Txt : {}\n# lbls: {}'.format(len(txt_lst),len(lbl_lst)))

## Label Distribution

d1={}
for i in lbl_lst:
    if i in d1:
        d1[i] += 1
    else:
        d1[i] = 1
print(d1)

df=pd.DataFrame()
df['label']=lbl_lst
df['Text_val']=txt_lst


X_train,X_test,y_train,y_test = train_test_split(df['Text_val'],df['label'],test_size=0.3,random_state=7)

print("X Train: {}, y_train: {}".format(X_train.shape,y_train.shape))
print("X_test: {}, y_test: {}".format(X_test.shape,y_test.shape))

## Tokenization

vocab_size = 1000
embeding_dims = 16
max_length = 16
trunc_type='post'
padding_type= 'post'
oov_token = '<OOV>'

tokenObj = Tokenizer( num_words= vocab_size, oov_token= oov_token)

tokenObj.fit_on_texts(X_train)
print('Word Index for Happy: {}'.format(tokenObj.word_index['happy']))

train_seq = tokenObj.texts_to_sequences(X_train)
padded_train_seq = pad_sequences(train_seq, maxlen= max_length, padding=padding_type, truncating=trunc_type)

test_seq = tokenObj.texts_to_sequences(X_test)
padded_test_seq = pad_sequences(test_seq, maxlen= max_length, padding=padding_type, truncating=trunc_type)

print("="*50)
print('X train Seq shape : {}, X test Seq Shape: {}'.format(padded_train_seq.shape,padded_test_seq.shape))

## Model

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

## Callbacks
def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
    ]
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = 'model_zero7.{epoch:02d}-{val_loss:.6f}.hdf5',
                               verbose=1,
                               save_best_only=True, save_weights_only = True)

max_epochs=30
history = model.fit(padded_train_seq,y_train, epochs= max_epochs, validation_data=(padded_test_seq,y_test)
                    #,callbacks=[get_callbacks(),reduce_lr,checkpointer]
)

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
plt.suptitle('Learning Curves')
plt.show()


## Embedding Layers
ymca=10
e = model.get_layer('embedding')
embed_wt= e.get_weights()
print('Embedding Weight: {}'.format(embed_wt[0].shape))

reverse_word_idx = {v:k for k,v in tokenObj.word_index.items()}


import io

out_v=io.open('vetc.tsv','w',encoding='utf-8')
out_m=io.open('meta.tsv','w',encoding='utf-8')

for i in range(1,vocab_size):
    word = reverse_word_idx[i]
    embed = embed_wt[0][i]

    out_m.write(word + "\n")
    out_v.write(  "\t".join([str(x) for x in embed]) + "\n")

out_m.close()
out_v.close()


## Model-2 ( Lets Try LSTM)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embeding_dims, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

max_epochs=30
history = model.fit(padded_train_seq,y_train, epochs= max_epochs, validation_data=(padded_test_seq,y_test)
                    #,callbacks=[get_callbacks(),reduce_lr,checkpointer]
)

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
plt.suptitle('Learning Curves')
plt.show()


## Model-3

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embeding_dims,input_length=max_length))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

max_epochs=30
history = model.fit(padded_train_seq,y_train, epochs= max_epochs, validation_data=(padded_test_seq,y_test)
                    #,callbacks=[get_callbacks(),reduce_lr,checkpointer]
)

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
plt.suptitle('Learning Curves')
plt.show()
