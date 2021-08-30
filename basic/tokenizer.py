import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


sentence=[
    "I love my dog!",
    "i, Love My cAt"
]

token_ins=Tokenizer(num_words=100)
token_ins.fit_on_texts(sentence)
print(token_ins.word_index)

sequences = token_ins.texts_to_sequences(sentence)

print(sequences)

print("="*50)
print("Try-2")


sentence_1=[
    "I love my dog!",
    "i, Love My cAt",
    "How are you doing?",
    "Are you alright"
]

sequences_1 = token_ins.texts_to_sequences(sentence_1)
print(sequences_1)

print("="*50)
print("Try-3 with OOV Token")

sentence=[
    "I love my dog!",
    "i, Love My cAt"
]

token_ins=Tokenizer(num_words=100,oov_token="<OOV>")
token_ins.fit_on_texts(sentence)
print(token_ins.word_index)

sequences = token_ins.texts_to_sequences(sentence)

print(sentence)
print(sequences)

print(sentence_1)
sequences_1 = token_ins.texts_to_sequences(sentence_1)
print(sequences_1)