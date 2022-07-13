import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Activation, UpSampling2D, Conv3D, Dense, Lambda, Input, Concatenate
from tensorflow.keras.optimizers import *
import tensorflow as tf
from tensorflow.keras import backend as K

HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)

class Brain(object):

    def __init__(self, state_size, action_size, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.dueling = arguments['dueling']
        self.optimizer_model = arguments['optimizer']
        self.model = self._build_model()
        #self.model_ = self._build_model()

    def _build_model(self):

        if self.dueling:
            x = Input(shape=(self.state_size,))

            # a series of fully connected layer for estimating V(s)

            y11 = Dense(self.num_nodes, activation='relu')(x)
            y12 = Dense(self.num_nodes, activation='relu')(y11)
            y13 = Dense(1, activation="linear")(y12)

            # a series of fully connected layer for estimating A(s,a)

            y21 = Dense(self.num_nodes, activation='relu')(x)
            y22 = Dense(self.num_nodes, activation='relu')(y21)
            y23 = Dense(self.action_size, activation="sigmoid")(y22)

            w = Concatenate(axis=-1)([y13, y23])

            # combine V(s) and A(s,a) to get Q(s,a)
            z = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                       output_shape=(self.action_size,))(w)
        else:
            #x = Input(shape=(self.state_size,))

            # a series of fully connected layer for estimating Q(s,a)

            #y1 = Dense(self.num_nodes, activation='relu')(x)
            #y2 = Dense(self.num_nodes, activation='relu')(y1)
            #z = Dense(self.action_size, activation="sigmoid")(y2)
            model = Sequential()
            model.add(Conv3D(16, kernel_size=3, strides=1, padding = 'same', input_shape=(1,10,10,3)))
            model.add(Activation("relu"))
            model.add(Conv3D(64, kernel_size=3, strides=1, padding = 'same'))
            model.add(Activation("relu"))
            model.add(Conv3D(256, kernel_size=3, strides=1, padding = 'same'))
            model.add(Activation("relu"))
            model.add(Flatten())
            model.add(Dense(self.action_size, activation='sigmoid'))
            model.summary()
            x = Input(shape=(1,10,10,3))
            y = model(x)

        model1 = Model(inputs=x, outputs=y)

        if self.optimizer_model == 'Adam':
            optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(lr=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')
        
        model1.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=optimizer)
        
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model1.load_weights(self.weight_backup)

        return model1

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output

        self.model.fit(x, y, batch_size=1, sample_weight=sample_weight, epochs=epochs, verbose=verbose)
        #self.model.train(x, y)

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.model_.predict(state)
        else:  # get prediction from local network
            return self.model.predict(state)

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1,1,10,10,3), target=target).flatten()

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save(self.weight_backup)
