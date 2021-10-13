import tensorflow as tf
print("Tf Version: {}".format(tf.version.VERSION))
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_lst=["I love my dog","I love my? cat","I love to be a cat what tdo you think of me"]
test_lst=["I love my durian","I love my? cat","I love to be a cat what tdo you think of me"]

token_obj=tf.keras.preprocessing.text.Tokenizer(num_words=100,oov_token="<OOV>")
token_obj.fit_on_texts(train_lst)

print(token_obj.word_index)


seq_train= token_obj.texts_to_sequences(train_lst)
seq_test= token_obj.texts_to_sequences(test_lst)

print(seq_train)
print(seq_test)

pad_train = pad_sequences(seq_train,maxlen=5,padding='post',truncating='post')
pad_test = pad_sequences(seq_test,maxlen=5,padding='post',truncating='post')

print("Paddes Train : {}".format(pad_train))
print("Paddes Test : {}".format(pad_test))