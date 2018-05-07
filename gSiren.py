import time
import logging
from gOptimizer import Optimizer
from nSiren import make
from tqdm import tqdm

def train_networks(networks):
    """Train each network.
    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))

    for network in networks:
        res = make.apply_async(kwargs=(network.network))
        # time.sleep(2)
        # print(network.network["nb_neurons"])
        network.loss = res.get()
        pbar.update(1)
    pbar.close()


def generate(generations, population, nn_param_choices, dataset):
    """Generate a network with the genetic algorithm.
    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))
        # Train and get accuracy for networks.
        train_networks(networks)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
            print(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.loss, reverse=False)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.
    Args:
        networks (list): The population of networks
    """
    logging.info('-' * 80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations = 3  # Number of times to evole the population.
    population = 8  # Number of networks in each generation.
    dataset = 'cifar10'

    nn_param_choices = {
        'dense': [64, 128, 256, 512, 768, 1024],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'lstm': [64, 128, 256, 512, 768, 1024],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'loss': ['binary_crossentropy'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
        'epochs': [1],
        'batch_size': [32, 64, 128, 256]
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main()