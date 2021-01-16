# Project name : Implementing Flappy bird game using Genetics Algorithm

## Team Members:
- Sahar Fakhrieh Kashan (960122680012)
------------


## Description:
this project is about AI learning how to play flappy bird with genetic algorithm.
it starts completely random, having no idea how to do and how the game will operate and after many generations, it starts to slowly learn and get better and figures out the patterns and how to progress further in the level.
it does all the works using a genetic algorithm named **NEAT**

------------
## What is NEAT:
NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm that creates artificial neural networks. For a detailed description of the algorithm.

Even if you just want to get the gist of the algorithm, reading at least a couple of the early NEAT papers is a good idea. Most of them are pretty short, and do a good job of explaining concepts (or at least pointing you to other references that will). The [initial NEAT paper](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf) is only 6 pages long, and Section II should be enough if you just want a high-level overview.

In the current implementation of NEAT-Python, a population of individual [genomes](https://neat-python.readthedocs.io/en/latest/glossary.html#term-genome) is maintained. Each genome contains two sets of [genes](https://neat-python.readthedocs.io/en/latest/glossary.html#term-gene)  that describe how to build an artificial neural network:

1. [Node](https://neat-python.readthedocs.io/en/latest/glossary.html#term-node) genes, each of which specifies a single neuron.
2. [Connection](https://neat-python.readthedocs.io/en/latest/glossary.html#term-connection) genes, each of which specifies a single connection between neurons.
To evolve a solution to a problem, the user must provide a fitness function which computes a single real number indicating the quality of an individual genome: better ability to solve the problem means a higher score. The algorithm progresses through a user-specified number of generations, with each generation being produced by reproduction (either sexual or asexual) and mutation of the most fit individuals of the previous generation.

for more informarion you can read [NEAT Overview](https://neat-python.readthedocs.io/en/latest/neat_overview.html)

------------

## Installation: 
for running this project you need to install : *NEAT* and *pygame *

for that you have to run these two commands in cmd :

>pip install pygame
>pip install neat-python



------------

## Instructions :
1.  install modules written in [installation](https://github.com/alitourani/computational-intelligence-class-9901/tree/master-1/G01-Genetic-Algorithm-Flappy-Bird#instructions-) section
2. Simply run flappy_bird.py and watch an AI start training itself to play the game of flappy bird!

------------

## Best way to understand :
first, play the [video](https://github.com/alitourani/computational-intelligence-class-9901/blob/master-1/G01-Genetic-Algorithm-Flappy-Bird/running%20flappy%20Bird%20.mp4)

second, read the[ pdf](https://github.com/alitourani/computational-intelligence-class-9901/blob/master-1/G01-Genetic-Algorithm-Flappy-Bird/flappy%20bird%20report.pdf)

and if you still have doubts look at the comments in the [source code](https://github.com/alitourani/computational-intelligence-class-9901/blob/master-1/G01-Genetic-Algorithm-Flappy-Bird/project/flappy_bird_source_code.py)

------------

## Photos of the game environment :
> When you start to run the game

![1](https://user-images.githubusercontent.com/71727363/104791728-09ca1a80-57b1-11eb-8f5a-6bfae69e6c6e.PNG)

>After a while (last seconds of the game)

![2](https://user-images.githubusercontent.com/71727363/104791838-5c0b3b80-57b1-11eb-8e88-0251be64e764.PNG)

>After the game

![3](https://user-images.githubusercontent.com/71727363/104791931-c623e080-57b1-11eb-9172-bcbce59716d2.PNG)
