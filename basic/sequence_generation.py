import tensorflow as tf

print("TF version: {}".format(tf.version.VERSION))
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "not available")

import tensorflow.keras.preprocessing.text
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

corpus = data.lower().split("\n")
print(corpus[:3])

tokenObj = tensorflow.keras.preprocessing.text.Tokenizer()
tokenObj.fit_on_texts(corpus)

# +1 for oov_word
vocab_size = len(tokenObj.word_index) + 1
print(vocab_size)

seq = tokenObj.texts_to_sequences(corpus)
print("{}===>{}".format(corpus[0], seq[0]))

input_sequences = []
cnt = 0
for each_seq in seq:
    # print(each_seq)
    start_pos = 1
    for inner_seq in each_seq:
        # print(each_seq[:start_pos+1])
        input_sequences.append(each_seq[:start_pos + 1])
        start_pos += 1
        if start_pos >= len(each_seq):
            break
    # cnt += 1
    # if cnt > 2:
    #     break

print(input_sequences[:8])

max_seq_length = max([len(each_seq) for each_seq in input_sequences])
print("Max Sequence Length: {}".format(max_seq_length))

pad_input_seq = np.array(pad_sequences(input_sequences,
                                       maxlen=max_seq_length,
                                       padding='pre'))

print(pad_input_seq[:8])

xs = pad_input_seq[:, :-1]
label = pad_input_seq[:, -1]

ys = to_categorical(label, num_classes=vocab_size)
print(ys)

## Model

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Embedding(vocab_size, 16, input_length=(max_seq_length - 1)))
model.add(tensorflow.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(xs, ys, epochs=500, verbose=0)

from matplotlib.pyplot import show, suptitle, subplots

fig, ax = subplots(1, 2, figsize=(12, 5))
epochs = range(len(history.history['accuracy']))

ax[0].plot(epochs, history.history['accuracy'], label='train', lw=2, marker='o', color='tomato')
# ax[0].plot(epochs, history.history['val_accuracy'], label='val', lw=2, marker='*', color='teal')
ax[0].set_title('Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()

ax[1].plot(epochs, history.history['loss'], label='train', lw=2, marker='o', color='tomato')
# ax[1].plot(epochs, history.history['val_loss'], label='val', lw=2, marker='o', color='teal')
ax[1].set_title('Loss')
ax[1].legend()

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
suptitle('Learning Curves')
show()


seed_text ="Laurance went to Dublin"
num_pred = 100

reverse_word_idx={v:k for k,v in tokenObj.word_index.items()}

for i in range(num_pred):
    a = pad_sequences(tokenObj.texts_to_sequences([seed_text]), maxlen=max_seq_length, padding='pre')
    pred_word = reverse_word_idx[np.argmax(model.predict(a))]
    seed_text = seed_text +" " + pred_word

print("Output==> {}".format(seed_text))