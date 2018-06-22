import random
from Demo import Neural_Network as nn
from Demo import Game as gm
import matplotlib.pyplot as plt
from Demo import Dynamic_Plot as dp
import time
import numpy as np


def genetic_algorithm():
    """ Generator function for genetic algorithm"""

    # Initial GA parameters
    population = 10
    generations = 500
    chromosome_length = 6

    # This of dynamic plotting
    plt.ion()
    dyn_plt = dp.DynamicUpdate()
    dyn_plt.on_launch()
    x_data = []
    y_data = []

    # Create a random initial agents of population size 10
    agents = initialize_agent(population, chromosome_length)

    # Generations
    for i in range(generations):

        print("Generation No: {}".format(i))

        # Evaluate fitness for each agent
        agents = fitness(agents, i)
        # Select agents
        agents = selection(agents)

        # This is for dynamic plotting graph generation vs fitness
        x_data.append(i)
        y_data.append(agents[0].fitness)
        dyn_plt.on_running(x_data, y_data)

        # Crossover and mutation
        agents = crossover(agents, population, chromosome_length)
        agents = mutation(agents, chromosome_length)

        # If any of the agent fitness reached 50 then stop. can be altered according to the needs
        if any(agent.fitness >= 50 for agent in agents):
            print("Reached {}% Fitness".format(agents[0].fitness))
            exit(0)


def initialize_agent(pop, length):
    """ Randomly initialize 10 Neural Network Agents """
    return [nn.NN(length) for _ in range(pop)]


def fitness(agents, gen):
    """ Calculate fitness - fitness value is obtained from Game.py. Calculate fitess for each agents"""
    agt_no = 0
    for agent in agents:
        agt_no += 1
        agent.fitness = gm.main(agent, gen, agt_no)

    return agents


def selection(agents):
    """ Select top 20% of fittest agents. Logic sort the agents according to its fitness values. """
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)
    print('\n'.join(map(str, agents)))
    agents = agents[:int(0.2 * len(agents))] # Top 20% of fittest agents

    return agents


def crossover(agents, pop, length):
    """ Crossover selected agents + top agents. Make size of 10 agent"""
    while pop > len(agents):

        # Parents
        par1 = random.choice(agents)
        par2 = random.choice(agents)

        # If two randomly selected agents are not equal then perform crossover.
        if par1 != par2:
            # Instantiate child agent
            child1 = nn.NN(length)
            child2 = nn.NN(length)

            # Crossover Index
            index = random.randint(0, length)
            # Child
            child1.chromosome_x = par1.chromosome_x[0:index] + par2.chromosome_x[index:length]
            child2.chromosome_x = par2.chromosome_x[0:index] + par1.chromosome_x[index:length]
            agents.extend([child1, child2])

    return agents


def mutation(agents, length):
    """ Mutate Chromosome at the mutation rate of 0.3. Select random gene then mutate the selected index"""
    mutation_rate = 0.3

    for agent in agents:

        for ind, char in enumerate(agent.chromosome_x):

            if random.uniform(0.0, 1.0) <= mutation_rate:
                # Mutation Logic
                agent.chromosome_x = agent.chromosome[0:ind] + \
                                     [float(agent.chromosome[ind]) + random.uniform(-2, 2)] + \
                                     agent.chromosome[ind+1:length]

    return agents


def plt_dynamic(x, y, ax, colors=['b']):
    """ This function is to plot generation vs fitness (dynamic plotting)"""
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()

if __name__ == "__main__":
    genetic_algorithm()
