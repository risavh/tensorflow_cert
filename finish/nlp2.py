import tensorflow as tf
print("TF Version: {}".format(tf.version.VERSION))
import  tensorflow_datasets as tfds


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

print(train_sentence[:2])
# token_obj = tf.keras.preprocessing.text.Tokenizer(num_words=10000,oov_token="<OOV>")
# token_obj.fit_on_texts(train_sentence)
#
# train_seq=token_obj.texts_to_sequences(train_sentence)
# test_seq=token_obj.texts_to_sequences(test_sentence)
#
# pad_seq_train=tf.keras.preprocessing.sequence.pad_sequences(train_seq,maxlen=30,truncating='post',padding='post')
# pad_seq_test=tf.keras.preprocessing.sequence.pad_sequences(test_seq,maxlen=30,truncating='post',padding='post')
#
#
# print(pad_seq_train[:2])
