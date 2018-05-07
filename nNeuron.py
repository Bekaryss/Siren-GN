from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint

class Neuron:

    def __init__(self, dense, dropout, lstm, activation, loss, optimizer, epochs, batch_size):
        self.dense = dense
        self.dropout = dropout
        self.lstm = lstm
        self.actication = activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.bach_size = batch_size


    def create_model(self, network_input, wights):
        model = Sequential()

        model.add(Dense(self.dense, input_shape=(network_input.shape[1], network_input.shape[2])))
        model.add(LSTM(self.lstm, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.dense))
        model.add(Dropout(self.dropout))
        model.add(Dense(network_input.shape[2]))
        model.add(Activation(self.actication))
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        if(wights != ''):
            model.load_weights(wights)

        return model

    def train_model(self, model, network_input, network_output):
        filepath = "weights_files/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        callbacks_list = []

        callbacks_list.append(ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'))

        model.fit(network_input, network_output, epochs=self.epochs, batch_size=self.bach_size, callbacks=callbacks_list, verbose=1)
        loss = model.evaluate(network_input, network_output, verbose=1)
        print(loss[0])
        return loss[0]