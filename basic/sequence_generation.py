import tensorflow as tf

print("TF version: {}".format(tf.version.VERSION))
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "not available")

import tensorflow.keras.preprocessing.text
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

pad_input_seq = pad_sequences(input_sequences,
                              maxlen=max_seq_length,
                              padding='pre')

print(pad_input_seq[:8])
