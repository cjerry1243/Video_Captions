from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, LSTM, Dense, Masking, Lambda, RepeatVector ,Merge,Concatenate, RepeatVector, Dropout
from keras.layers import TimeDistributed, Permute, Dot, Activation, Reshape
from keras.utils import to_categorical
import numpy as np
from load_data import load_train


def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_categorical_crossentropy(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply categorical_crossentropy with the mask
        loss = K.categorical_crossentropy(y_true, y_pred) * mask
        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked_categorical_crossentropy


def seq2seq(latent_dim, dim, vocabs, pad):
    print('Build model...')
    # Encoder
    encoder_inputs = Input(shape=( 80, dim))
    encoder = LSTM(latent_dim, return_state=True, return_sequences=False, name='encoder')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [encoder_outputs,state_h, state_c]

    encoder_model = Model(encoder_inputs, encoder_states)

    '''''''''
    # Attention
    attention = TimeDistributed(Dense(max_sentence, activation='sigmoid'),
                                name = 'attention')(encoder_states[0])      # (80, max_sentence)
    attention = Permute([2,1])(attention)                                   # (max_sentence, 80)
    attention = Activation('softmax')(attention)
    reshape = Reshape([80,latent_dim])(encoder_states[0])                   # (80, latent_dim)
    repeat = Dot([2,1])([attention, reshape])                               # (max_sentence, latent_dim)
    '''
    repeat = RepeatVector(1)(encoder_outputs)

    # Decoder
    # decoder_inputs = Input(shape=(max_sentence, vocabs))
    # mask = Masking(pad)(decoder_inputs)
    # mask = Concatenate(axis=-1)([repeat,mask])

    decoder_lstm = LSTM(latent_dim, return_sequences=True ,name='decoder')
    decoder_dense = Dense(vocabs, activation='softmax', name='dense')

    # decoder_outputs = decoder_lstm(mask, initial_state=[encoder_states[1],encoder_states[2]])
    # decoder_outputs = decoder_dense(decoder_outputs)

    decoder_inputs = Input(shape=(1, vocabs))

    all_outputs = []
    inputs = Concatenate(axis=-1)([repeat,decoder_inputs])
    for _ in range(max_sentence):
        # Run the decoder on one timestep
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        outputs = decoder_dense(outputs)
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)



    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model, encoder_model

def cap_generator(X, Y_teach, Y, batch_size):
    # batch_x = np.empty([batch_size, 80, 4096])
    batch_y_teach = np.empty([batch_size, max_sentence, vocabs])
    batch_y = np.empty([batch_size, max_sentence, vocabs])
    shuffle = np.random.permutation(train_samples)

    end = 0
    while True:
        start = end if end < train_samples else 0
        end = min(start + batch_size, train_samples)
        index = range(start,end)#shuffle[] # np.random.randint(train_samples, size=batch_size)
        #print(index[0:5])
        batch_x = X[index]
        for i,ind in enumerate(index):
            y_ind = 1 # np.random.randint(Y_teach[ind].shape[0]-2)
            batch_y_teach[i] = to_categorical(Y_teach[ind][y_ind],vocabs)
            batch_y[i] = to_categorical(Y[ind][y_ind],vocabs)
            # batch_y_teach[i] = to_categorical(Y_teach[ind][-1],vocabs)
            # batch_y[i] = to_categorical(Y[ind][-1],vocabs)
        yield [batch_x,batch_y_teach],batch_y

def val_generator(X, Y_teach, Y, batch_size):
    # batch_x = np.empty([batch_size, 80, 4096])
    batch_y_teach = np.empty([batch_size, max_sentence, vocabs])
    batch_y = np.empty([batch_size, max_sentence, vocabs])
    while True:
        index = train_samples + np.arange(batch_size) # np.random.randint(val_samples, size=batch_size)
        #print(index[0:5])
        batch_x = X[index]
        for i,ind in enumerate(index):
            y_ind = 2 #np.random.randint(Y_teach[ind].shape[0]-2)
            batch_y_teach[i] = to_categorical(Y_teach[ind][y_ind],vocabs)
            batch_y[i] = to_categorical(Y[ind][y_ind],vocabs)
        #     batch_y_teach[i] = to_categorical(Y_teach[ind][-1],vocabs)
        #     batch_y[i] = to_categorical(Y[ind][-1],vocabs)
        yield [batch_x,batch_y_teach],batch_y

def s2vt(latent_dim):
    #s2vt
    encoder_model = Sequential()
    encoder_model.add(LSTM(latent_dim, return_sequences=True, input_shape=(None, dim)))
    encoder_model.add(TimeDistributed(Dense(1)))

    decoder_model = Sequential()
    #decoder_model.add(Masking())
    decoder_model.add(TimeDistributed(Dense(1),input_shape=(None, vocabs)))

    model = Sequential()
    model.add(Merge([encoder_model, decoder_model],mode='concat'))
    model.add(LSTM(latent_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(vocabs, activation='softmax')))

    return model


# model parameters:
vocabs = 3000
max_sentence = 40
batch_size = 50
latent_dim = 512
dim = 4096
samples = 1450
train_samples = 1450
val_samples = samples- train_samples

if __name__ == '__main__':
    ### Load data
    xx, xx_decoder, yy, EOS= load_train(max_sentence, vocabs)
    xx = xx.reshape(samples*80,dim)
    xx = (xx-xx.mean(axis=0))/(xx.std(axis=0)+0.001)
    xx = xx.reshape(samples,80,dim)



    EOS = to_categorical(EOS, vocabs)
    pad = to_categorical(0, vocabs)
    masked_categorical_crossentropy = get_loss(pad)
    generator = cap_generator(xx, xx_decoder, yy, batch_size)
    generator_val = val_generator(xx,xx_decoder,yy, val_samples)

    model, encoder = seq2seq(latent_dim, dim, vocabs, pad)
    print(model.summary())
    #model = s2vt(latent_dim)
    print('Train...')
    model.compile(optimizer='adam', loss=masked_categorical_crossentropy, metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='seq2seq_cp_weights_nonteach.h5', verbose=0, monitor='loss',
                                   save_best_only=True, save_weights_only= True)

    model.fit_generator(generator, epochs=200, steps_per_epoch=train_samples/batch_size, callbacks=[checkpointer])#,
                        #validation_data=generator_val, validation_steps=1)
    model.save_weights('seq2seq_weights.h5')