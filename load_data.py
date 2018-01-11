import json
import numpy as np
from vocab_dictionary import Vocabs_Tokenizer



def load_train(maxlen, vocabs):
    print('need numpy==1.13 to use np.unique()')
    ### maxlen: max timesteps, vocabs:
    tokenizer, _ = Vocabs_Tokenizer(vocabs)

    print('Loading training data......')
    train_featpath = 'MLDS_hw2_data/training_data/feat/'
    ### load training_label.json
    f = open('MLDS_hw2_data/training_label.json','r')
    train_json = json.load(f)
    f.close()
    ### train_json[index]['caption','id']

    x_train = np.zeros([len(train_json), 80, 4096])
    decoder_dicty_train = {}
    dicty_train = {}

    for i in range(len(train_json)):
        #print(i)
        id = train_json[i]['id']
        # xxx = np.load(train_featpath + id + '.npy')
        x_train[i] = np.load(train_featpath + id + '.npy') # (xxx-xxx.mean(axis = 0))/(xxx.std(axis=0)+0.001)


        ### Load y
        captions = train_json[i]['caption']
        decoder_y_temp = np.zeros([len(captions), maxlen])
        y_temp = np.zeros([len(captions), maxlen])
        for j,sentence in enumerate(captions):
            sentence_out = sentence + ' EOS'
            sentence_in = 'BOS ' + sentence
            output = tokenizer.texts_to_sequences([sentence_out])
            input = tokenizer.texts_to_sequences([sentence_in])

            if len(output[0])> maxlen:
                decoder_y_temp[j, :] = input[0][:maxlen]
                y_temp[j, :] = output[0][:maxlen]
            else:
                decoder_y_temp[j, :len(input[0])] = input[0]
                y_temp[j, :len(output[0])] = output[0]
        decoder_dicty_train[i] = np.unique(decoder_y_temp, axis=0)
        dicty_train[i] = np.unique(y_temp, axis=0)
    EOS = tokenizer.texts_to_sequences(['EOS'])
    return x_train, decoder_dicty_train, dicty_train, EOS


def load_test():
    test_featpath = 'MLDS_hw2_data/testing_data/feat/'
    ### load testing_label.json
    f = open('MLDS_hw2_data/testing_label.json','r')
    test_json = json.load(f)
    f.close()
    ### test_json[index]['caption','id']

    x_test = np.zeros([len(test_json),80,4096])
    answer ={}
    index_id = {}
    id_index = {}
    for i in range(len(test_json)):
        id = test_json[i]['id']
        # xxx = np.load(test_featpath + id + '.npy')

        x_test[i] = np.load(test_featpath + id + '.npy') # (xxx-xxx.mean(axis = 0))/(xxx.std(axis=0)+0.001)

        captions = test_json[i]['caption']
        answer[id] = captions
        index_id[i] = id
        id_index[id] = i
    return x_test, answer, index_id, id_index


if __name__ == '__main__':
    vocabs = 3000
    max_sentence = 50
    x_train, decoder_dicty_train, dicty_train, EOS= load_train(max_sentence, vocabs)

    #np.save('x_train.npy',x_train)
    #np.save('decoder_train_inputs.npy',decoder_inputy_train)
    #np.save('y_train.npy',y_train)
