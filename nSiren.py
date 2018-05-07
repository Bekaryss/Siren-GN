import random

from nMidiUtils import MidiUtils
from nNeuron import Neuron
import glob
import os
from celery import Celery
app = Celery('task', backend='rpc://', broker='amqp://beka:beka@192.168.1.47:5672/vhost_beka')

midiUtil = MidiUtils('midi_songs/*.mid', 30, 100, 60, 0.25, 'output.mid')
neuron = Neuron(128, 0.1, 128, 'sigmoid', 'binary_crossentropy', "rmsprop", 4, 64)

@app.task
def make(dense, dropout, lstm, activation, loss, optimizer, epochs, batch_size):
    print("Worker")
    loss = train(dense, dropout, lstm, activation, loss, optimizer, epochs, batch_size)
    return loss

def train(dense, dropout, lstm, activation, loss, optimizer, epochs, batch_size):
    neuron = Neuron(dense, dropout, lstm, activation, loss, optimizer, epochs, batch_size)
    # list_of_files = glob.glob('weights_files/*')  # * means all if need specific format then *.csv
    latest_file = ''
    # if list_of_files:
    #     latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    input_network, output_network = midiUtil.preprocessing()
    print(input_network.shape[0], input_network.shape[1], input_network.shape[2])
    model = neuron.create_model(input_network, latest_file)
    loss = neuron.train_model(model, input_network, output_network)
    return loss


# if __name__ == '__main__':
#     train()