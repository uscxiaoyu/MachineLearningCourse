import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import networkx as nx

class GeneticAlgorithm:
    TOUR_SIZE = 2  # the number of individuals selected for tournament competitions 
    def __init__(self, fitne_func, indiv_bound, muta_rate=0.25, cros_rate=0.8, num_popul=200):
        self.muta_rate = muta_rate
        self.cros_rate = cros_rate
        self.num_popul = num_popul
        self.indiv_bound = indiv_bound
        self.indiv_dim = len(self.indiv_bound)
        self.fitne_func = fitne_func
        self.fitne_values = []
        self.last_iter = {"population": [], "fitness": []}
        self.population = []
        self.best_solu = []
        for _ in range(self.num_popul):
            self.population.append([(x[1] - x[0])*np.random.rand() + x[0] for x in self.indiv_bound])
            
    def eval_fitness(self):
        self.fitne_values = []
        for indiv in self.population:
            if indiv in self.last_iter["population"]:
                idx = self.last_iter["population"].index(indiv)
                self.fitne_values.append(self.last_iter["fitness"][idx])
            else:
                self.fitne_values.append(self.fitne_func(indiv))
    
    def selection(self):
        '''
        Implementation of binary tournament selection
        '''
        candi_father = np.random.randint(low=0, high=self.num_popul, size=self.TOUR_SIZE)
        candi_mother = np.random.randint(low=0, high=self.num_popul, size=self.TOUR_SIZE)
        return (np.argmin([self.fitne_values[i] for i in candi_father]),
                np.argmin([self.fitne_values[i] for i in candi_mother]))
    
    def crossover(self, i, j):
        '''
        Interchange gene of two selected individuals according to the probability of crossover
        '''
        father, mother = self.population[i], self.population[j]
        if np.random.rand() < self.cros_rate:
            div_dim = np.random.randint(low=0, high=self.indiv_dim-1)  # choose the gene
            indiv_1 = father[:div_dim+1] + mother[div_dim+1:]
            indiv_2 = mother[:div_dim+1] + father[div_dim+1:]
            return indiv_1, indiv_2
        
        return father, mother
 
    def mutation(self, indiv):
        '''
        Change the gene of an individual according the probability of mutation
        '''
        if np.random.rand() < self.muta_rate:
            dim = np.random.randint(low=0, high=self.indiv_dim)  # choose one gene to mutate
            indiv[dim] = (self.indiv_bound[dim][1] - self.indiv_bound[dim][0])*np.random.rand() + self.indiv_bound[dim][0]
        
        return indiv
        
    def reproduction(self):
        '''
        Generate new generation: including selection, crossover and mutation
        '''
        new_generation = []
        new_fitne_values = []
        while len(new_generation) < self.num_popul:
            i, j = self.selection()
            indiv_1, indiv_2 = self.crossover(i, j)
            self.mutation(indiv_1)
            self.mutation(indiv_2)
            
            if indiv_1 not in new_generation:
                new_generation.append(indiv_1)
            if indiv_2 not in new_generation:
                new_generation.append(indiv_1)
        
        self.population = new_generation
        self.fitne_values = new_fitne_values
    
    def evolution(self, num_generation=10, is_print=False):
        v = 0
        while True:
            self.eval_fitness()
            best_solu_idx = np.argmin(self.fitne_values)
            self.best_solu.append([self.fitne_values[best_solu_idx], self.population[best_solu_idx]])
            if is_print:
                print(f"Generation {v}, the best fitness value is {self.fitne_values[best_solu_idx]}, the best individual is{self.population[best_solu_idx]}")
            v += 1
            if v == num_generation:
                break
            
            self.last_iter = {'population': self.population[:], 'fitness': self.fitne_values[:]}  # record the last iteration infos
            self.reproduction()

if __name__ == "__main__":
    import time
    def f(x):
        
    