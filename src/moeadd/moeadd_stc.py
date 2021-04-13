# -*- coding: utf-8 -*-

import numpy as np
from src.moeadd.moeadd import *
from src.moeadd.moeadd_supplementary import *
from copy import deepcopy
from abc import ABC, abstractproperty, abstractmethod
import matplotlib.pyplot as plt

class moeadd_solution(ABC):
    '''
    
    Abstract superclass of the moeadd solution. *__hash__* method must be declared in the subclasses. 
    Overloaded *__eq__* method of moeadd_solution uses strict equatlity between self.vals attributes,
    therefore, can not be used with the real-valued strings.
    
    '''
    def __init__(self, x, obj_funs):
        self.vals = x
        self.obj_funs = obj_funs
        self.precomputed_value = False
        self.precomputed_domain = False
    
    @property
    def obj_fun(self):
        if self.precomputed_value: 
            return self._obj_fun
        else:
            self._obj_fun = np.fromiter(map(lambda obj_fun: obj_fun(self.vals), self.obj_funs), dtype = float)
            self.precomputed_value = True
            return self._obj_fun

    def get_domain(self, weights):
        if self.precomputed_domain:
            return self._domain
        else:
            self._domain = get_domain_idx(self, weights)
            self.precomputed_domain = True
            return self._domain
    
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.vals == other.vals
        else:
            return NotImplemented
        
    def __call__(self):
        return self.obj_fun
    
    @abstractmethod
    def __hash__(self):
        raise NotImplementedError('The hash needs to be defined in the subclass')


class moe_population_constructor(ABC):
    '''
    
    Abstract class of the creator of new moeadd solutions, utilized in its initialization phase. 
    Shall be overwritten to be properly used.
    
    Methods:
    ---------
    
    __init__(*args) 
        In the __init__ method, you should be able to set the parameters of the constructor.
        
    create(*creation_args)
        Method, dedicated to the creation (oftenly randmized) of new candidate solutions.
        
    Example of the subclass:
        >>> class test_population_constructor(object):
        >>>     def __init__(self, bitstring_len = 2, vals_range = [-4, 4]):
        >>>         self.bs_len = bitstring_len; self.vals_range = vals_range
        >>>         
        >>>     def create(self, *args):
        >>>         created_solution = solution_array(x = np.random.uniform(low = self.vals_range[0], 
        >>>                                                                 high = self.vals_range[1], 
        >>>                                                                 size = self.bs_len), 
        >>>                              obj_funs=[optimized_fun_1, optimized_fun_2])
        >>>         return created_solution        
    -------
    '''
    def __init__(self, *constr_args):
        pass
        
    @abstractmethod
    def create(self, *creation_args):
        return None


class moe_evolutionary_operator(ABC):
    '''
    
    Abstract class of the moeadd evolutionary operator. The subclass implementations shall
    have mutation and crossover methods, that produce correspondingly one new solution and 
    a list of new solutions;
    
    Methods:
    -------
        
    mutation(solution)
        return a new solution, created by alteration of an existing one
        
    crossover(parents_pool)
        returns a list of new solutions, created from the parents pool. Parents pool already 
        constains the selected individuals, therefore, no new selection required.
    
    '''
    def __init__(self, xover_lambda, mut_lambda):
        pass

    @abstractmethod
    def mutation(self, solution):
        return None

    @abstractmethod
    def crossover(self, parents_pool):
        return [None for parent in parents_pool]
    

def gaussian_mutation(solution_x):
    '''
        Basic Gaussian mutation, that can be used inside the moeadd evolutionary operator, 
        when it works with string of real values. More complicated ones can be declared in its image. 
        
        Arguments:
        ---------
        
        solution_x : np.array, 
            values (genotype) of the moeadd solution. Represented by the *moeadd_solution.vals* attribute or the 
            same attribute of its subclass object.        
            
    '''
    return solution_x + np.random.normal(size = solution_x.size)


def mixing_xover(parents):
    '''
        Basic crossover operator, that can be used inside the moeadd evolutionary operator, 
        when it works with string of real values. More complicated ones can be declared in its image. 
        
        Arguments:
        ---------
        
        parents : list of 2 moeadd_solution, or its subclass objects, 
            parent solutions of the many-objective optimization algorithm.

        Returns:
        --------
        
        offsprings : list of 2 moeadd_solution, or its subclass objects,
            offspring solutions of the many-objective optimization algorithm, 
            with values, creating in the interval between their parent ones.            

    '''

    proportion = np.random.uniform(low = 1e-6, high = 0.5-1e-6)
    offsprings = [deepcopy(parent) for parent in parents]
    offsprings[0].precomputed_value = False; offsprings[1].precomputed_value = False
    offsprings[0].precomputed_domain = False; offsprings[1].precomputed_domain = False

    offsprings[0].vals = parents[0].vals + proportion * (parents[1].vals - parents[0].vals)
    offsprings[1].vals = parents[0].vals + (1 - proportion) * (parents[1].vals - parents[0].vals)
    return offsprings

def simple_selector(sorted_neighbors, number_of_neighbors = 4):
    '''
        Simple selector of neighboring weight vectors: takes n-closest (*n = number_of_neighbors*)ones to the 
        processed one. Defined to be used inside the moeadd algorithm.
    
        Arguments:
        ----------
        
        sorted_neighbors : list
            proximity list of neighboring vectors, ranged in the ascending order of the angles between vectors.
            
        number_of_neighbors : int
            numbers of vectors to be considered as the adjacent ones
            
        Returns:
        ---------
        
        sorted_neighbors[:number_of_neighbors] : list
            self evident slice of proximity list
    '''
    return sorted_neighbors[:number_of_neighbors]


def plot_pareto(levels, weights = None, max_level = None):
    '''
    
    Vizualization method to demonstrate the pareto levels of 2D-problem on the plane via matplotlib
    
    Arguments:
    ----------
    
    levels : src.moeadd.pareto_levels obj
        object, containing pareto levels. Ususally obtained from *src.moeadd.moeadd_optimizer* attribute *src.moeadd.moeadd_optimizer.pareto_levels*.
        
    weights : np.ndarray, optional
        contains weights from the moeadd algorithm to be visualized
    
    '''
    if max_level is None:
        max_level = np.inf
    else:
        max_level += 1
    print([front_idx for front_idx in np.arange(min((len(levels.levels), max_level)))])
    coords = [[(solution.obj_fun[0], solution.obj_fun[1]) for solution in levels.levels[front_idx]] for front_idx in np.arange(min((len(levels.levels), max_level)))]
    coords_arrays = []
    for coord_set in coords:
        coords_arrays.append(np.array(coord_set))
    coords_arrays
    colors = ['r', 'r', 'b', 'y', 'g'] + ['m' for idx in np.arange(len(coords_arrays) - 5)]
    fig, ax = plt.subplots()
    for front_idx in np.arange(len(coords_arrays)):
        ax.plot([1.04862679e-18, 1.29891486e-14, 1.30104261e-11, 0.00401406], [6, 2, 1, 0], color = 'r', linewidth = 1)
        ax.scatter(coords_arrays[front_idx][:, 0], coords_arrays[front_idx][:, 1], color = colors[front_idx], s = 5)
        plt.xlabel('Quality')
        plt.ylabel('Complexity')
#        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim((1e-19, 0.006))
        plt.grid()
    if not (weights is None):
        for weight_idx in np.arange(weights.shape[0]):
            vector_coors = weights[weight_idx, :]
            ax.plot([0, vector_coors[0]], [0, vector_coors[1]], color = 'k')
    fig.show()