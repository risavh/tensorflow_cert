import tensorflow as tf
import tensorflow.keras.preprocessing.text
import numpy as np
print("TF Version: {}".format(tf.version.VERSION))
import json

path=r"D:\works\koglu\datasets\sarcasm\archive (1)\Sarcasm_Headlines_Dataset_v2.json"


f = open(path,'r')
data=f.readlines()
f.close()
print(len(data),data[:3])


labels=[]
headline=[]

cnt=0
for i in data:
    #print(i,type(dict(i)))
    row_wise = json.loads(i)
    #print(i["is_sarcastic"])
    labels.append(row_wise['is_sarcastic'])
    headline.append(row_wise['headline'])
    # if cnt>3:
    #     break
    # cnt+=1

# print(headline)

print("Length Headline: {}".format(len(headline)))

train_headline = headline[0:2000]
test_headline = headline[2000:]

train_label = np.array(labels[0:2000])
test_label = np.array(labels[2000:])



vocab_size=10000
maxlen = 7
embed_dim=16

token_obj = tensorflow.keras.preprocessing.text.Tokenizer(num_words=vocab_size,oov_token="<OOV>")

token_obj.fit_on_texts(train_headline)


train_seq = token_obj.texts_to_sequences(train_headline)
test_seq = token_obj.texts_to_sequences(test_headline)

pad_train_seq= tensorflow.keras.preprocessing.sequence.pad_sequences(train_seq,maxlen=maxlen,padding='post',
                                                                     truncating='post')
pad_test_seq= tensorflow.keras.preprocessing.sequence.pad_sequences(test_seq,maxlen=maxlen,padding='post',
                                                                     truncating='post')
print(pad_train_seq[:2])
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Embedding(vocab_size,embed_dim,input_length=maxlen))
model.add(tensorflow.keras.layers.GlobalAveragePooling1D())
model.add(tensorflow.keras.layers.Dense(6,activation='relu'))
model.add(tensorflow.keras.layers.Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(pad_train_seq,train_label,epochs=7,validation_data=(pad_test_seq,test_label))

model.evaluate(pad_test_seq,test_label)[1]