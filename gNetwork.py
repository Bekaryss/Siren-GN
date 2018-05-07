"""Class that represents the network to be evolved."""
import random
import logging

class Network:
        def __init__(self, nn_param_choices=None):
            self.loss = 100
            self.nn_param_choices = nn_param_choices
            self.network = {}

        def create_random(self):
            """Create a random network."""
            for key in self.nn_param_choices:
                self.network[key] = random.choice(self.nn_param_choices[key])

        def create_set(self, network):
            """Set network properties.
            Args:
                network (dict): The network parameters
            """
            self.network = network

        def train(self):
            """Train the network and record the accuracy.
            Args:
                dataset (str): Name of dataset to use.
            """
            # neuron = Neuron(self.network['dense'],
            #                 self.network['dropout'],
            #                 self.network['lstm'],
            #                 self.network['activation'],
            #                 self.network['loss'],
            #                 self.network['optimizer'],
            #                 self.network['epochs'],
            #                 self.network['batch_size'],)
            #
            # return neuron

        def print_network(self):
            """Print out a network."""
            print(self.network)
            print("Network loss: %.2f%%" % (self.loss))
            logging.info(self.network)
            logging.info("Network loss: %.2f%%" % (self.loss))



# class Neuron:
#
#     def __init__(self, dense, dropout, lstm, activation, loss, optimizer, epochs, batch_size):
#         self.dense = dense
#         self.dropout = dropout
#         self.lstm = lstm
#         self.actication = activation
#         self.loss = loss
#         self.optimizer = optimizer
#         self.epochs = epochs
#         self.bach_size = batch_size

