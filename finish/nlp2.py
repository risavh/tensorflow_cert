import tensorflow as tf
print("TF Version: {}".format(tf.version.VERSION))
import  tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']
print(train_data,type(train_data))

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

vocab_size =10000
max_length=120
embeding_dim=16
print(train_sentence[:2])
token_obj = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,oov_token="<OOV>")
token_obj.fit_on_texts(train_sentence)

train_seq=token_obj.texts_to_sequences(train_sentence)
test_seq=token_obj.texts_to_sequences(test_sentence)

pad_seq_train=tf.keras.preprocessing.sequence.pad_sequences(train_seq,maxlen=120,truncating='post',padding='post')
pad_seq_test=tf.keras.preprocessing.sequence.pad_sequences(test_seq,maxlen=120,truncating='post',padding='post')

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


print(pad_seq_train[:2])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,embeding_dim,input_length=max_length))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(6,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='adam',metrics=['accuracy'])

model.fit(pad_seq_train,train_labels,epochs=5,validation_data=(pad_seq_test,test_labels))

model.evaluate((pad_seq_test,test_labels))

