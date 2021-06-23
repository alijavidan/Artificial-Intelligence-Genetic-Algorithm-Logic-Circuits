import random
import pandas as pd
import operator
import pandas as pd
import multiprocessing
from functools import partial
import numpy as np
import time
import math
import copy

chromosome_size = 9
pop_size = 10
mutationRate = (1/pop_size + 1/chromosome_size)/2
crossoverRate = 0.78
nPoint = 2
selectionRoom = int(pop_size*0.2)
randomChromosomeShare = 0.1

class Chromosome:
    def __init__(self):
        self.genomes = []
        self.fittness = 0
        self.probability = None
        self.rank = None

    def eval(self, arguments):
        input = arguments[0]

        output = input
        for i in range(1, len(arguments)-1):
            for gate in [genome.gate for genome in self.genomes]:
                    output = gate.eval(output, arguments[i])

        return output

class Genome:
    def __init__(self, gate):
        self.gate = gate

class Gate:
    pass

class AND(Gate):
    def eval(self, input1, input2):
        return input1 and input2

class OR(Gate):
    def eval(self, input1, input2):
        return input1 or input2

class XOR(Gate):
    def eval(self, input1, input2): 
        return input1 ^ input2

class NAND(Gate):
    def eval(self, input1, input2):
        return not(input1 and input2)

class NOR(Gate):
    def eval(self, input1, input2):
        return not(input1 or input2)

class XNOR(Gate):
    def eval(self, input1, input2):
        if input1 == input2:
            return True
        else:
            return False

def generateRandomGate():
    rand = random.randint(1, 6)

    if rand == 1:
        return AND()
    elif rand == 2:
        return OR()
    elif rand == 3:
        return XOR()
    elif rand == 4:
        return NAND()
    elif rand == 5:
        return NOR()
    elif rand == 6:
        return XNOR()

def createInitialPopulation():
    global pop_size
    chromosomes = []

    while len(chromosomes) < pop_size:
        chromosome = Chromosome()
        while len(chromosome.genomes) < chromosome_size:
            chromosome.genomes.append(Genome(generateRandomGate()))

        chromosomes.append(chromosome)

    return chromosomes

def evaluateChromosome(chromosome, chunk):
    for index, row in chunk.iterrows():
        if  row.iloc[-1] == chromosome.eval(row.values.tolist()):
            chromosome.fittness+=1

def evaluatePopulation(population, chunk):
    for chromosome in population:
        evaluateChromosome(chromosome, chunk)
    
    return population

def onePointCrossover(selected):
    offspring = []

    while len(offspring) < pop_size:
        for i in range(0, len(selected), 2):
            if random.random() < crossoverRate:
                rand = random.randint(0, chromosome_size-1)

                new_chromosome = Chromosome()
                new_chromosome.genomes = selected[i].genomes[0:rand] + selected[i+1].genomes[rand:len(selected[i+1].genomes)]
                offspring.append(new_chromosome)

                new_chromosome = Chromosome()
                new_chromosome.genomes = selected[i+1].genomes[0:rand] + selected[i].genomes[rand:len(selected[i].genomes)]
                offspring.append(new_chromosome)          
            else:
                new_chromosome = Chromosome()
                new_chromosome.genomes = selected[i].genomes
                offspring.append(new_chromosome)
                new_chromosome = Chromosome()
                new_chromosome.genomes = selected[i+1].genomes
                offspring.append(new_chromosome)

    return offspring

def mutation(offspring):
    for chromosome in offspring:
        for genome in chromosome.genomes:
            if random.random() < mutationRate:
                randomGate = generateRandomGate()
                while type(randomGate) == type(genome.gate):
                    randomGate = generateRandomGate()
                genome.gate = randomGate

def nPointCrossover(selected):
    global nPoint
    offspring = []

    while len(offspring) < pop_size:
        for i in range(0, len(selected), 2):
            if random.random() < crossoverRate:
                points = []
                for k in range(nPoint):
                    point = random.randint(0, chromosome_size-1)
                    while point in points:
                        point = random.randint(0, chromosome_size-1)
                    points.append(random.randint(0,chromosome_size-1))

                points.sort()

                previous_point = 0
                new_chromosome_1 = Chromosome()
                new_chromosome_2 = Chromosome()
                for j in range(len(points)):
                    if j % 2 == 0:
                        new_chromosome_1.genomes += selected[i].genomes[previous_point:points[j]]
                        new_chromosome_2.genomes += selected[i+1].genomes[previous_point:points[j]]
                    else:
                        new_chromosome_1.genomes += selected[i+1].genomes[previous_point:points[j]]
                        new_chromosome_2.genomes += selected[i].genomes[previous_point:points[j]]
                    previous_point = points[j]
                if len(points) % 2 == 0:
                    new_chromosome_1.genomes += selected[i].genomes[previous_point:chromosome_size]
                    new_chromosome_2.genomes += selected[i+1].genomes[previous_point:chromosome_size]
                else:
                    new_chromosome_1.genomes += selected[i+1].genomes[previous_point:chromosome_size]
                    new_chromosome_2.genomes += selected[i].genomes[previous_point:chromosome_size]

                offspring.append(new_chromosome_1)
                offspring.append(new_chromosome_2)
            else:
                new_chromosome = Chromosome()
                new_chromosome.genomes = selected[i].genomes
                offspring.append(new_chromosome)
                new_chromosome = Chromosome()
                new_chromosome.genomes = selected[i+1].genomes
                offspring.append(new_chromosome)

    return offspring

# def selectFittest(population): 
    #     return population[0:selectionRoom]

def rouletteWheelSelection(population):
    sum_of_fittnesses = 0
    previous_probability = 0

    for chromosome in population:
        sum_of_fittnesses += chromosome.fittness

    for chromosome in population:
        chromosome.probability = previous_probability + chromosome.fittness/sum_of_fittnesses
        previous_probability = chromosome.probability

    population.sort(key=operator.attrgetter("probability"))
    selected = []
    while len(selected) < selectionRoom*(1-randomChromosomeShare):
        r = random.random()
        previous_probability = 0
        for i in range(len(population)):
            if r > previous_probability and r <= population[i].probability:
                if not population[i] in selected: 
                    selected.append(population[i])
                break
                
            previous_probability = population[i].probability

    addRandomChromosomeToMatingPool(selected)
                
    return selected

def rankSelection(population):
    population.sort(key=operator.attrgetter("fittness"))
    for i in range(len(population)):
        population[i].rank = i+1

    selected = []
    while len(selected) < selectionRoom*(1-randomChromosomeShare):
        r = random.random()
        previous_rank = 0
        for i in range(len(population)):
            if r > previous_rank/len(population) and r <= population[i].rank/len(population):
                if not population[i] in selected: 
                    selected.append(population[i])
                break

            previous_rank = population[i].rank
    
    addRandomChromosomeToMatingPool(selected)
                
    return selected

def generateRandomSuccessor(chromosome):
    successor = copy.deepcopy(chromosome)
    successor.fittness = 0

    random_genome = random.randint(0, chromosome_size-1)
    randomGate = generateRandomGate()
    while type(randomGate) == type(chromosome.genomes[random_genome].gate):
        randomGate = generateRandomGate()
    successor.genomes[random_genome].gate = randomGate

    return successor

def simulatedAnnealing(population, truth_table):
    for chromosome in population:
        randomSuccessor = generateRandomSuccessor(chromosome)
        evaluateChromosome(randomSuccessor, truth_table)
        deltaE = randomSuccessor.fittness - chromosome.fittness
        if deltaE > 0:
            chromosome = randomSuccessor
        else:
            probabiliy = pow(math.e, deltaE/T)
            if random.random() < probabiliy:
                chromosome = randomSuccessor

def addRandomChromosomeToMatingPool(selected): 
        while len(selected) < selectionRoom:
            chromosome = Chromosome()
            while len(chromosome.genomes) < chromosome_size:
                chromosome.genomes.append(Genome(generateRandomGate()))

def stopCondition(population, len):
    if len in [chromosome.fittness for chromosome in population]:
        printSolution(chromosome)
        return True
    
    return False

def printSolution(chromosome):
    for genome in chromosome.genomes:
        print(type(genome))

#Main
if __name__ == '__main__':    
    truth_table = pd.read_csv("truth_table.csv")
    truth_table_length =  len(truth_table.index)
    population = createInitialPopulation()
    while True:
        num_processes = multiprocessing.cpu_count() - 1
        chunk_size = int(truth_table.shape[0]/num_processes)
        chunks = np.array_split(truth_table, num_processes)
        pool = multiprocessing.Pool(processes=num_processes)
        prod_x = partial(evaluatePopulation, population)
        results = pool.map(prod_x, chunks)
        for chromosome in population:
            for result in results:
                chromosome.fittness += result[population.index(chromosome)].fittness

        if stopCondition(population, truth_table_length):
            print('Finish')
            break
        average = sum([chromosome.fittness for chromosome in population])/len(population)
        maximum = max([chromosome.fittness for chromosome in population])
        print('average: ', average, ',   max: ', maximum)
        T = truth_table_length - average
        simulatedAnnealing(population, truth_table)
        # selected = selectFittest(population)
        selected = rouletteWheelSelection(population)
        # selected = rankSelection(population)
        offspring = onePointCrossover(selected)
        # offspring = nPointCrossover(selected)
        population = offspring
