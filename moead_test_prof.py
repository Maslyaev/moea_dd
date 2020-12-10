#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:06:44 2020

@author: mike_ubuntu
"""

import numpy as np
from src.moeadd import *
from src.moeadd_supplementary import *
from copy import deepcopy

import matplotlib.pyplot as plt

seed = 2
np.random.seed(seed)


class test_population_constructor(object):
    def __init__(self, bitstring_len = 2, vals_range = [-4, 4]):
        self.bs_len = bitstring_len; self.vals_range = vals_range
        
    def create(self, *args):
        created_solution = solution_obj(x = np.random.uniform(low = self.vals_range[0], high = self.vals_range[1], size = self.bs_len), 
                             obj_funs=[optimized_fun_1, optimized_fun_2])
        return created_solution        

    
class test_evolutionary_operator(object):
    def __init__(self, xover_lambda, mut_lambda):
        self._xover = xover_lambda
        self._mut_lambda = mut_lambda
        
    def mutation(self, solution):
        output = deepcopy(solution)
        output.x_vals = self._mut_lambda(output.x_vals) #output.x_vals + np.random.normal(scale = )
        return output

    def crossover(self, parents_pool):
        offspring_pool = []
        for idx in np.arange(np.int(np.floor(len(parents_pool)/2.))):
            print(parents_pool[2*idx].x_vals, parents_pool[2*idx+1].x_vals)
            offsprings = self._xover((parents_pool[2*idx], parents_pool[2*idx+1]))
            offspring_pool.extend(offsprings)
        return offspring_pool


def plot_pareto(levels, weights):
    coords = [[(solution.obj_fun[0], solution.obj_fun[1]) for solution in levels.levels[front_idx]] for front_idx in np.arange(len(levels.levels))]
    coords_arrays = []
    for coord_set in coords:
        coords_arrays.append(np.array(coord_set))
    coords_arrays
    colors = ['r', 'k', 'b', 'y', 'g'] + ['m' for idx in np.arange(len(coords_arrays) - 5)]
    fig, ax = plt.subplots()
    for front_idx in np.arange(len(coords_arrays)):
        ax.scatter(coords_arrays[front_idx][:, 0], coords_arrays[front_idx][:, 1], color = colors[front_idx])
    for weight_idx in np.arange(weights.shape[0]):
        vector_coors = weights[weight_idx, :]
        ax.plot([0, vector_coors[0]], [0, vector_coors[1]], color = 'k')
    fig.show()
#        ax.plot([0, 1], [0, 1], color = 'k')
    

def optimized_fun_1_heavy(x, x_size_sqrt):
    return 1 - np.exp(- np.sum((x - 1/x_size_sqrt)**2))

def optimized_fun_2_heavy(x, x_size_sqrt):
    return 1 - np.exp(- np.sum((x + 1/x_size_sqrt)**2))

def optimized_fun_1(x, x_size_sqrt):
    return 1 - np.exp(- (x[0] - 1/x_size_sqrt)**2 - (x[1] - 1/x_size_sqrt)**2) #- np.sum((x - 1/x_size_sqrt)**2))

def optimized_fun_2(x, x_size_sqrt):
    return 1 - np.exp(- (x[0] + 1/x_size_sqrt)**2 - (x[1] + 1/x_size_sqrt)**2) #- np.sum((x - 1/x_size_sqrt)**2))



pop_constr = test_population_constructor()
optimizer = moeadd_optimizer_constrained(pop_constr, 20, 100, [optimized_fun_1, optimized_fun_2], None, delta = 1/50., neighbors_number = 5)

constr_1_1 = Inequality(lambda x: x[0] + 4)
constr_1_2 = Inequality(lambda x: x[1] + 4)
constr_2_1 = Inequality(lambda x:  - x[0] + 4)
constr_2_2 = Inequality(lambda x:  - x[1] + 4)

optimizer.set_constraints(constr_1_1, constr_1_2, constr_2_1, constr_2_2)

gaussian_mutation = lambda solution_x: solution_x + np.random.normal(size = solution_x.size)
#mixing_xover = lambda parents: parents[1].obj_

def mixing_xover(parents):
    proportion = np.random.uniform(low = 1e-6, high = 0.5-1e-6)
    offsprings = [deepcopy(parent) for parent in parents]
    offsprings[0].precomputed_value = False; offsprings[1].precomputed_value = False
    offsprings[0].precomputed_domain = False; offsprings[1].precomputed_domain = False
    
    offsprings[0].x_vals = parents[0].x_vals + proportion * (parents[1].x_vals - parents[0].x_vals)
    offsprings[1].x_vals = parents[0].x_vals + (1 - proportion) * (parents[1].x_vals - parents[0].x_vals)
    return offsprings
    
operator = test_evolutionary_operator(mixing_xover, gaussian_mutation)
optimizer.set_evolutionary(operator=operator)

optimizer.pass_best_objectives(0, 0)
def simple_selector(sorted_neighbors, number_of_neighbors = 4):
    return sorted_neighbors[:number_of_neighbors]

optimizer.optimize(simple_selector, 0.95, (4,), 50, 0.75)