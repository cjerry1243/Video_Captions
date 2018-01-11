from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, Lambda, Merge, Concatenate, RepeatVector
from keras.layers import TimeDistributed, Permute, Activation, Dot, Reshape
from keras.utils import to_categorical
import numpy as np
from load_data import load_test
from vocab_dictionary import Vocabs_Tokenizer
from model_seq2seq_1 import seq2seq, dim, vocabs, latent_dim, max_sentence

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

tokenizer, reverse_tokenizer = Vocabs_Tokenizer(vocabs)
reverse_tokenizer[0]='<pad>'
pad = to_categorical(0, vocabs)

EOS_num = tokenizer.texts_to_sequences(['eos'])[0][0]
### decoder input in the inference model
BOS = to_categorical(tokenizer.texts_to_sequences(['BOS']), vocabs)
BOS = BOS.reshape(1, 1, vocabs)



def attention_inference(latent_dim, max_sentence):
    # Attention
    encoder_output = Input(shape=(80, latent_dim))
    attention = TimeDistributed(Dense(max_sentence, activation='sigmoid'),
                                name = 'attention')(encoder_output)
    attention = Permute([2,1])(attention)
    attention = Activation('softmax')(attention)
    reshape = Reshape([80,latent_dim])(encoder_output)
    attention_step = Dot([2,1])([attention, reshape])
    model = Model(encoder_output, attention_step)

    return model

def decoder_model(latent_dim, pad):
    decoder_inputs = Input(shape=(1, vocabs))
    encoder_outputs = Input(shape=(latent_dim,))
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [encoder_outputs, decoder_state_input_h, decoder_state_input_c]

    encoder_outputs = RepeatVector(1)(encoder_outputs)


    mask = Masking(pad)(decoder_inputs)
    mask = Concatenate(axis=-1)([encoder_outputs,mask])


    decoder_lstm = LSTM(latent_dim, return_sequences=True,return_state=True ,name='decoder')
    decoder_outputs, state_h, state_c = decoder_lstm(
        mask, initial_state=[decoder_states_inputs[1], decoder_states_inputs[2]])
    decoder_states = [state_h, state_c]

    decoder_dense = Dense(vocabs, activation='softmax', name='dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return decoder


def decode_sequence(input_seq, encoder, attention, decoder):
    # Encode the input as state vectors.
    states_value = encoder.predict(input_seq)
    step = 0
    sequence_p = np.ones([beam,])
    total_p = np.zeros([beam,])
    tree_p = np.ones([beam, beam])
    # word_index = np.zeros([beam,])
    word_index = np.zeros([beam,])
    tree_index = np.zeros([beam, beam])
    sequence_index = np.empty([beam,0])
    total_index = [[]]* beam
    # attention_out = attention.predict(states_value[0])
    # print(attention_out.shape)

    # print(states_value[0])
    # Generate empty target sequence of length 1.
    target_seq = [BOS]* beam
    states_value = [states_value]* beam
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    update = [1]*beam
    stop_condition = False

    while not stop_condition:
        for i in range(beam):
            if update[i] == 0:
                continue
            # states_value[0] = attention_out[0, step : step+1]
            #print(step,states_value[0].shape)
            output_tokens, h, c = decoder.predict([target_seq[i]] + states_value[i])
            states_value[i] = [states_value[i][0], h, c]
            # Sample a token
            next_word_index = (-output_tokens[0, -1, :]).argsort()[:beam] # np.argmax(output_tokens[0, -1, :])
            tree_index[i] = np.array(next_word_index)
            word_p = output_tokens[0, -1][next_word_index]
            # print(step, word_p)
            for j in range(beam):
                tree_p[i,j] = sequence_p[i]*word_p[j]
        # print(step, sequence_p)
        # print(tree_p)
        if step == 0:
            maxindex_p = k_largest_index_argsort(tree_p[0:1], beam)
            tree_p_0 = tree_p
        else:
            tree_p_0 = np.zeros([beam, beam])
            for k in range(beam):
                if update[k] == 1:
                    tree_p_0[k] = tree_p[k]
            maxindex_p = k_largest_index_argsort(tree_p_0, beam)

        for k in range(beam):
            if update[k] == 0:
                continue
            ind = maxindex_p[k]
            sequence_p[k] = tree_p_0[ind[0], ind[1]]
            word_index[k] = tree_index[ind[0], ind[1]]
            target_seq[k] = np.zeros((1, 1, vocabs))
            target_seq[k][0, 0, int(word_index[k])] = 1.
            states_value[k] = states_value[ind[0]]
        # print(step, word_index, sequence_p)
        if step == 0:
            sequence_index = np.column_stack((sequence_index, word_index))
        else:
            for k in range(beam):
                if update[k] == 0:
                    continue
                ind = int(maxindex_p[k,0])
                sequence_index[k] = sequence_index[ind]
            sequence_index = np.column_stack((sequence_index, word_index))
            # exit()
        for k, token_num in enumerate(sequence_index[:,-1]):
            # print(step, total_p[n], token_num, word_p[n])
            if total_p[k] == 0. and token_num == EOS_num:
                total_index[k] = sequence_index[k]
                total_p[k] = sequence_p[k]
                update[k] = 0

        if sequence_index.shape[1] > max_sentence-1:
            for k in range(beam):
                if total_p[k] == 0.:
                    total_index[k] = sequence_index[k]
                    total_p[k] = sequence_p[k]
            stop_condition = True
        if sum(update) == 0:
            stop_condition = True
        # print(sequence_index, total_index)
        step = step + 1

    for t in range(beam):
        total_p[t] = total_p[t]**(1/(len(total_index[t])))
        decoded_sentence = []
        for token in total_index[t]:
            sampled_char = reverse_tokenizer[token]
            decoded_sentence.append(sampled_char)
        decoded_sentence = trim(decoded_sentence)
        print(' '.join([w for w in decoded_sentence]), total_p[t])

    max_search = total_p.argmax()
    decoded_sentence = []
    for token in total_index[max_search]:
        sampled_char = reverse_tokenizer[token]
        decoded_sentence.append(sampled_char)
    # print(' '.join([w for w in decoded_sentence]))
    return decoded_sentence


def trim(test_output):
    for i,w in enumerate(test_output):
        if w == 'eos':
            test_output = test_output[:i]
            break
    for j in range(len(test_output)):
        if test_output[-1] == 'eos':
            del test_output[-1]
        elif test_output[-1] == '<pad>':
            del test_output[-1]
        else:
            break
    return test_output

beam = 1
samples = 100
if __name__ == '__main__':
    mmm, encoder = seq2seq(latent_dim, dim, vocabs, pad)
    encoder.load_weights('seq2seq_cp_weights.h5', by_name= True)

    decoder = decoder_model(latent_dim, pad)
    decoder.load_weights('seq2seq_cp_weights.h5', by_name= True)

    attention = attention_inference(latent_dim, max_sentence)
    # attention.load_weights('seq2seq_cp_weights.h5', by_name=True)

    # print(encoder.summary(), decoder.summary())
    x_test, test_answer, index_id, id_index = load_test()
    x_test = x_test.reshape(samples * 80, dim)
    x_test = (x_test - x_test.mean(axis=0)) / (x_test.std(axis=0) + 0.001)
    x_test = x_test.reshape(samples, 80, dim)


    output_filename = 'beam_output_sentences.txt'
    ff = open(output_filename, 'w')
    f = open('MLDS_hw2_data/testing_id.txt', 'r')

    for id in f.readlines():
        id = id.strip()
        i = id_index[id]
        test_output = decode_sequence(x_test[i:i+1], encoder, attention, decoder)
        test_output = trim(test_output)

        print(id,' '.join([w for w in test_output]))
        ff.write(id+','+' '.join([w for w in test_output])+'.\n')
    f.close()
    ff.close()
