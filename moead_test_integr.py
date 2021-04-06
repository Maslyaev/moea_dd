#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:06:44 2020

@author: mike_ubuntu
"""

import numpy as np
from src.moeadd.moeadd import *
from src.moeadd.moeadd_supplementary import *
from src.moeadd.moeadd_stc import *
from copy import deepcopy

import matplotlib.pyplot as plt

seed = 1488
np.random.seed(seed)

class solution_array(moeadd_solution):
    def __init__(self, x, obj_funs):
        super().__init__(x, obj_funs)
        self.x_size_sqrt = np.sqrt(self.vals.size)
#        self.epsilon = np.full(shape = 2, fill_value = 1e-9)
    
    @property
    def obj_fun(self):
        if self.precomputed_value: 
            return self._obj_fun
        else:
            self._obj_fun = np.fromiter(map(lambda obj_fun: obj_fun(self.vals, self.x_size_sqrt), self.obj_funs), dtype = float)
            self.precomputed_value = True
            return self._obj_fun

    def __eq__(self, other):
        if isinstance(other, type(self)):
            epsilon = 1e-9
            return all([abs(self.vals[0] - other.vals[0]) < epsilon,
                        abs(self.vals[1] - other.vals[1]) < epsilon]) #np.all(np.abs(self.vals - other.vals) < self.epsilon)
        else:
            return NotImplemented
    
    def __hash__(self):
        return hash(tuple(self.vals))

#    @property
#    def latex_form(self):
##        return r'$S(\vec{{u}})=\begin{{array}}{{cc}} L_1(\vec{{u}})=0  \\ ... \\ L_k(\vec{{u}})=0 \end{{array}}.$'
#        form = (r"\begin{eqnarray*} |\nabla\phi| &=& 1,\\ \frac{\partial \phi}{\partial t} + U|\nabla \phi| &=& 0 "
#                r"\end{eqnarray*}")
#        return form

class test_population_constructor(moe_population_constructor):
    def __init__(self, bitstring_len = 2, vals_range = [-4, 4]):
        self.bs_len = bitstring_len; self.vals_range = vals_range
        
    def create(self, *args):
        created_solution = solution_array(x = np.random.uniform(low = self.vals_range[0], high = self.vals_range[1], size = self.bs_len), 
                             obj_funs=[optimized_fun_1, optimized_fun_2])
        return created_solution        

    
class test_evolutionary_operator(moe_evolutionary_operator):
    def __init__(self, xover_lambda, mut_lambda):
        self._xover = xover_lambda
        self._mut_lambda = mut_lambda
        
    def mutation(self, solution):
        output = deepcopy(solution)
        output.vals = self._mut_lambda(output.vals) #output.vals + np.random.normal(scale = )
        return output

    def crossover(self, parents_pool):
        offspring_pool = []
        for idx in np.arange(np.int(np.floor(len(parents_pool)/2.))):
#            print(parents_pool[2*idx].vals, parents_pool[2*idx+1].vals)
            offsprings = self._xover((parents_pool[2*idx], parents_pool[2*idx+1]))
            offspring_pool.extend(offsprings)
        return offspring_pool
    

def optimized_fun_1_heavy(x, x_size_sqrt):
    return 1 - np.exp(- np.sum((x - 1/x_size_sqrt)**2))

def optimized_fun_2_heavy(x, x_size_sqrt):
    return 1 - np.exp(- np.sum((x + 1/x_size_sqrt)**2))

def optimized_fun_1(x, x_size_sqrt):
    return 1 - np.exp(- (x[0] - 1/x_size_sqrt)**2 - (x[1] - 1/x_size_sqrt)**2) #- np.sum((x - 1/x_size_sqrt)**2))

def optimized_fun_2(x, x_size_sqrt):
    return 1 - np.exp(- (x[0] + 1/x_size_sqrt)**2 - (x[1] + 1/x_size_sqrt)**2) #- np.sum((x - 1/x_size_sqrt)**2))


pop_constr = test_population_constructor()
optimizer = moeadd_optimizer_constrained(pop_constr, 20, 100, None, delta = 1/50., neighbors_number = 5)

constr_1_1 = Inequality(lambda x: x[0] + 4)
constr_1_2 = Inequality(lambda x: x[1] + 4)
constr_2_1 = Inequality(lambda x:  - x[0] + 4)
constr_2_2 = Inequality(lambda x:  - x[1] + 4)

optimizer.set_constraints(constr_1_1, constr_1_2, constr_2_1, constr_2_2)

    
operator = test_evolutionary_operator(mixing_xover, gaussian_mutation)
optimizer.set_evolutionary(operator=operator)

optimizer.pass_best_objectives(0, 0)


optimizer.optimize(simple_selector, 0.95, (4,), 100, 0.75)