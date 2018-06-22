import random
import numpy as np


class NN:

    base = [0.41702648, 1.3604004, 0.38370492, -0.14096331, -3.31637945, -1.96085111]
    # Base value to start from - this is taken from experimentation.
    # Otherwise it may take several generation to reach this point.

    def __init__(self, chromosome_length, hidden_node=2, hidden_layers=1, learning_rate=0.3):

        self.learning_rate = learning_rate
        self.chromosome_length = chromosome_length
        self.fitness = 0
        self.chromosome = self.random()

        self.input_node = 2  # number of input nodes 2
        self.op_node = 1  # for binary classifier required output node should be 1.
        self.hidden_node = hidden_node  # number of hidden nodes in each layer
        self.hidden_layers = hidden_layers  # number of hidden layers ie. 1

        self.train_x = None
        self.w1_ly1 = None
        self.w2_ly2 = None
        self.b1_ly1 = None
        self.b2_ly2 = None

    def random(self):
        """ Random weights around base for initialize population stage in GA"""
        gene = []
        for weight in self.base:
            gene.append(weight + random.triangular(weight - self.learning_rate, weight + self.learning_rate))

        return gene

    @property
    def chromosome_x(self):
        """ Getter method to get chromosome for GA"""
        return self.chromosome

    @chromosome_x.setter
    def chromosome_x(self, ch_new):
        """ Setter method to crossover and mutation in GA"""
        self.chromosome = ch_new

    def __str__(self):
        """ Display the chromosome and its fitness value """
        return 'String: ' + str(self.chromosome) + ' Fitness: ' + str(self.fitness)

    @staticmethod
    def activation(z):
        """ Sigmoid Activation function """
        return 1 / (1 + np.exp(-z))

    def feed_forward(self, train_x):

        """Feed forward network to generate paddle controller value"""

        # Reshape the chromosome as required in Neural Network.
        # First 4 gene is for layer 1 to hidden layer weight and the rest for hidden to output layer.
        self.w1_ly1 = np.asarray(list(map(float, self.chromosome[0:4]))).reshape(self.input_node, self.hidden_node)
        self.w2_ly2 = np.asarray(list(map(float, self.chromosome[4:len(self.chromosome)])))\
            .reshape(self.hidden_node, self.op_node)
        # Bias of 1 for hidden layer and output layer
        self.b1_ly1 = np.ones((1, self.hidden_node))
        self.b2_ly2 = np.ones((1, self.op_node))

        # Features for NN from Game.py (Input to Neural network)
        train_x = np.asarray(train_x).reshape(1, self.input_node)

        # Forward Propogation
        a1 = train_x
        z2 = np.dot(a1, self.w1_ly1) + self.b1_ly1  # z = Wx + b
        a2 = self.activation(z2)

        z3 = np.dot(a2, self.w2_ly2) + self.b2_ly2
        a3 = self.activation(z3)

        return a3[0][0]  # For Controlling the paddle position.
