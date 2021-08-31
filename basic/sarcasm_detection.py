import tensorflow as tf
print("TF Version: {}".format(tf.version.VERSION))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json
from tqdm import tqdm

path=r"D:\works\koglu\DL_Training\sarcasm\archive (1)\Sarcasm_Headlines_Dataset_v2.json"

f=open(path,"r")
data=f.readlines()
print(type(data),len(data))

lst_label=[]
lst_data=[]
for idx,i in tqdm(enumerate(data)):
    #print("idx: {} has  {} type {}".format(idx,i,type(json.loads(i))))
    dict_val=json.loads(i)
    lst_label.append(dict_val['is_sarcastic'])
    lst_data.append(dict_val['headline'])
    # if idx>1:
    #     break

print("# X Train: {}, # y train: {}".format(len(lst_data),len(lst_label)))

print(lst_data[:3],lst_label[:3])

token_obj = Tokenizer(num_words=10000, oov_token="<OOV>")

token_obj.fit_on_texts(lst_data[:3])

seq_obj=token_obj.texts_to_sequences(lst_data[:3])
pad_seq_obj=pad_sequences(seq_obj,padding='post',truncating='post',maxlen=30)

print(token_obj.word_index)
print(seq_obj)
print(pad_seq_obj)