from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Lambda, Input, Concatenate
from tensorflow.keras.optimizers import SGD

class Brain(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  
        self.model = self._build_model()              
        
    def _build_model(self):

        model = Sequential()
        model.add(Dense(12, input_dim = self.state_size, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(Dense(self.action_size, activation = 'sigmoid'))
        opt = SGD(lr = 0.01, momentum = 0.9)
        model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
        return model
        
    def train(self, x, y, sample_weight=None, epochs=10, verbose=0):
        history = self.model.fit(x,y,epochs = 500, verbose = 0)
        _,train_acc = self.model.evaluate(x, y, verbose = 0)
        print(train_acc,"cost")
        
    def predict_one_sample(self, state, target=False):
        return self.model.predict(state)
