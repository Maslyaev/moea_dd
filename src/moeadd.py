#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:29:18 2020

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
from functools import reduce
from src.moeadd_supplementary import fast_non_dominated_sorting, slow_non_dominated_sorting, NDL_update, Equality, Inequality, acute_angle

class solution_obj(object):
    def __init__(self, x, obj_funs):
        self.x_vals = x
        self.obj_funs = obj_funs
        self.x_size_sqrt = np.sqrt(self.x_vals.size)
        self.precomputed_value = False
        self.precomputed_domain = False
    
    @property
    def obj_fun(self):
        if self.precomputed_value: 
            return self._obj_fun
        else:
            self._obj_fun = np.fromiter(map(lambda obj_fun: obj_fun(self.x_vals, self.x_size_sqrt), self.obj_funs), dtype = float)
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
        epsilon = 1e-9
#        print('Checking the equality')
        if isinstance(other, solution_obj):
            return np.all(np.abs(self.x_vals - other.x_vals) < epsilon)
        else:
            return NotImplemented
        
    def __call__(self):
        return self.obj_fun
    
    def __hash__(self):
        return hash(tuple(self.x_vals))
        
        
#def default_mating_selection(weights, neighborhood, population, offspring_pool_size = 2, close_proximity_probability = 0.95):
#    if np.random.uniform() < close_proximity_probability:
#        candidate_parents_idxs = np.random.choice(neighborhood[1:])
#        parent_region = [weights][0] # ------ Дописать --------
#    else:
#        return np.random.choice(population, size = offspring_pool_size, replace = False)


def get_domain_idx(solution, weights) -> int:
    if type(solution) == np.ndarray or type(solution) == np.ndarray:
        return np.argmin(np.array(list(map(lambda x: acute_angle(x, solution), weights))))
    elif type(solution.obj_fun) == np.ndarray:
        return np.argmin(np.fromiter(map(lambda x: acute_angle(x, solution.obj_fun), weights), dtype = float))
    else:
        raise ValueError('Can not detect the vector of objective function for solution')
    

def penalty_based_intersection(sol_obj, weight, ideal_obj, penalty_factor = 1) -> float:
    d_1 = np.dot((sol_obj.obj_fun - ideal_obj), weight) / np.linalg.norm(weight)
    d_2 = np.linalg.norm(sol_obj.obj_fun - (ideal_obj + d_1 * weight/np.linalg.norm(weight)))
    return d_1 + penalty_factor * d_2


def population_to_sectors(population, weights): # Много жрёт
    solution_selection = lambda weight_idx: [solution for solution in population if solution.get_domain(weights) == weight_idx]
    return list(map(solution_selection, np.arange(len(weights))))    


def clear_list_of_lists(inp_list) -> list:
    return [elem for elem in inp_list if len(elem) > 0]

    
class pareto_levels(object):
    def __init__(self, population, sorting_method = fast_non_dominated_sorting, update_method = NDL_update):
        self._sorting_method = sorting_method
        self.population = population
        self._update_method = update_method
        self.levels = self._sorting_method(self.population)
        
    def sort(self):
        self.levels = self._sorting_method(self.population)
    
    def update(self, point):
#        print('Update to add point', point.x_vals)
#        print('Pareto levels update: before', [solution.x_vals for solution in  self.population])
#        print('lengths:', len(self.population), sum([len(level) for level in self.levels]))        
#        for level_idx, level in enumerate(self.levels):
#            print(level_idx, [solution.x_vals for solution in level])
        self.levels = self._update_method(point, self.levels)
        self.population.append(point)
#        print('\n')
#        print('-||- after', [solution.x_vals for solution in  self.population])
#        print('lengths:', len(self.population), sum([len(level) for level in self.levels]))
#        for level_idx, level in enumerate(self.levels):
#            print(level_idx, [solution.x_vals for solution in level])
#        print('\n')
#        print('\n')
#    
    def delete_point(self, point):  # Разобраться с удалением.  Потенциально ошибка
#        print('deleting', point.x_vals)
        new_levels = []
        for level in self.levels:
            temp = deepcopy(level)
            if point in temp: temp.remove(point)
            if not len(temp) == 0: new_levels.append(temp)
#        print(point, point.x_vals, type(point), '\n')
#        print('population x_vals:', [solution.x_vals for solution in self.population], '\n')
#        print('population objects:', [solution for solution in self.population], '\n')        
        population_cleared = []

        for elem in self.population:
            if not elem == point: population_cleared.append(elem)
        print('population:', len(population_cleared), 'levels:', sum([len(level) for level in new_levels]))
        if len(population_cleared) != sum([len(level) for level in new_levels]):
            print('initial population', [solution.x_vals for solution in self.population],'\n')
            print('cleared population', [solution.x_vals for solution in population_cleared],'\n')
            print(point.x_vals)
            raise Exception('Deleted something extra')
        self.levels = new_levels
        self.population = population_cleared
#        self.population.remove(point)
            

def locate_pareto_worst(levels, weights, best_obj, penalty_factor = 1.):
    domain_solutions = population_to_sectors(levels.population, weights)
    most_crowded_count = np.max([len(domain) for domain in domain_solutions]); crowded_domains = [domain_idx for domain_idx in np.arange(len(weights)) if 
                                                                           len(domain_solutions[domain_idx]) == most_crowded_count]
    if len(crowded_domains) == 1:
        most_crowded_domain = crowded_domains[0]
    else:
        PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, weights[domain_idx], best_obj, penalty_factor) for sol_obj in domain_solutions[domain_idx]])
        PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
        most_crowded_domain = crowded_domains[np.argmax(PBIS)]
        
    worst_NDL_section = []
    domain_solution_NDL_idxs = np.empty(most_crowded_count)
    for solution_idx, solution in enumerate(domain_solutions[most_crowded_domain]):
        domain_solution_NDL_idxs[solution_idx] = [level_idx for level_idx in np.arange(len(levels.levels)) 
                                                    if np.any([solution == level_solution for level_solution in levels.levels[level_idx]])][0]
        
    max_level = np.max(domain_solution_NDL_idxs)
    worst_NDL_section = [domain_solutions[most_crowded_domain][sol_idx] for sol_idx in np.arange(len(domain_solutions[most_crowded_domain])) 
                        if domain_solution_NDL_idxs[sol_idx] == max_level]
    PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, weights[most_crowded_domain], best_obj, penalty_factor), worst_NDL_section), dtype = float)
    return worst_NDL_section[np.argmax(PBIS)]        


class moeadd_optimizer(object):
    '''
    
    Solving multiobjective optimization problem (minimizing set of functions)
    
    '''
    def __init__(self, pop_constructor, weights_num, pop_size, optimized_functionals, solution_params, delta, neighbors_number, 
                 NDS_method = fast_non_dominated_sorting, NDL_update = NDL_update):
        population = []
        for solution_idx in range(pop_size):
            while True:
                temp_solution = pop_constructor.create(solution_params)
                if not np.any([temp_solution == solution for solution in population]):
                    population.append(temp_solution)
                    break
        self.pareto_levels = pareto_levels(population, sorting_method=NDS_method, update_method=NDL_update)
        
        self.opt_functionals = optimized_functionals
        self.weights = []; weights_size = len(optimized_functionals) #np.empty((pop_size, len(optimized_functionals)))
        for weights_idx in range(weights_num):
            while True:
                temp_weights = self.weights_generation(weights_size, delta)
                if not temp_weights in self.weights:
                    self.weights.append(temp_weights)
                    break
        self.weights = np.array(self.weights)

        self.neighborhood_lists = []
        for weights_idx in range(weights_num):
            self.neighborhood_lists.append([elem_idx for elem_idx, _ in sorted(
                    list(zip(np.arange(weights_num), [np.linalg.norm(self.weights[weights_idx, :] - self.weights[weights_idx_inner, :]) for weights_idx_inner in np.arange(weights_num)])), 
                    key = lambda pair: pair[1])][:neighbors_number+1]) # срез листа - задаёт регион "близости"

        self.best_obj = None

        
    @staticmethod
    def weights_generation(weights_num, delta) -> list:
        weights = np.empty(weights_num)
        assert 1./delta == round(1./delta) # check, if 1/delta is integer number
        m = np.zeros(weights_num)
        for weight_idx in np.arange(weights_num):
            weights[weight_idx] = np.random.choice([div_idx * delta for div_idx in np.arange(1./delta + 1 - np.sum(m[:weight_idx + 1]))])
            m[weight_idx] = weights[weight_idx]/delta
        weights[-1] = 1 - np.sum(weights[:-1])
        assert (weights[-1] <= 1 and weights[-1] >= 0)
        return list(weights) # Переделать, т.к. костыль
    
        
    def pass_best_objectives(self, *args) -> None:
        assert len(args) == len(self.opt_functionals)
        self.best_obj = np.empty(len(self.opt_functionals))
        for arg_idx, arg in enumerate(args):
            self.best_obj[arg_idx] = arg if isinstance(arg, int) or isinstance(arg, float) else arg() # Переделать под больше elif'ов
    
    
    
    def set_evolutionary(self, operator) -> None:
        # добавить возможность теста оператора
        self.evolutionary_operator = operator
    
    
    @staticmethod
    def mating_selection(weight_idx, weights, neighborhood_vectors, population, neighborhood_selector, neighborhood_selector_params, delta) -> list:
        parents_number = int(len(population)/4.) # Странное упрощение   
        if np.random.uniform() < delta:
            selected_regions_idxs = neighborhood_selector(neighborhood_vectors[weight_idx], *neighborhood_selector_params)
            candidate_solution_domains = list(map(lambda x: x.get_domain(weights), [candidate for candidate in population]))

            solution_mask = [(population[solution_idx].get_domain(weights) in selected_regions_idxs) for solution_idx in candidate_solution_domains]
            available_in_proximity = sum(solution_mask)
            parent_idxs = np.random.choice([idx for idx in np.arange(len(population)) if solution_mask[idx]], 
                                            size = min(available_in_proximity, parents_number), 
                                            replace = False)
            if available_in_proximity < parents_number:
                parent_idxs_additional = np.random.choice([idx for idx in np.arange(len(population)) if not solution_mask[idx]], 
                                            size = parents_number - available_in_proximity, 
                                            replace = False)
                parent_idxs_temp = np.empty(shape = parent_idxs.size + parent_idxs_additional.size)
                parent_idxs_temp[:parent_idxs.size] = parent_idxs; parent_idxs_temp[parent_idxs.size:] = parent_idxs_additional
                parent_idxs = parent_idxs_temp
        else:
            parent_idxs = np.random.choice(np.arange(len(population)), size = parents_number, replace = False)
        return parent_idxs
    
    
    def update_population(self, offspring, PBI_penalty):
        '''
        Update population to get the pareto-nondomiated levels with the worst element removed. 
        Here, "worst" means the solution with highest PBI value (penalty-based boundary intersection)
        '''
#        domain = get_domain_idx(offspring, self.weights)        
        
        self.pareto_levels.update(offspring)  #levels_updated = NDL_update(offspring, levels)
        if len(self.pareto_levels.levels) == 1:
            worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)
        else:
            if self.pareto_levels.levels[len(self.pareto_levels.levels) - 1] == 1:
                domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
                reference_solution = self.pareto_levels.levels[len(self.pareto_levels.levels) - 1][0]
                reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
                if len(domain_solutions[reference_solution_domain] == 1):
                    worst_solution = locate_pareto_worst(self.pareto_levels.levels, self.weights, self.best_obj, PBI_penalty)                            
                else:
                    worst_solution = reference_solution
            else:
                last_level_by_domains = population_to_sectors(self.pareto_levels.levels[len(self.pareto_levels.levels)-1], self.weights)
                most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
                crowded_domains = [domain_idx for domain_idx in np.arange(len(self.weights)) if len(last_level_by_domains[domain_idx]) == most_crowded_count]

                if len(crowded_domains) == 1:
                    most_crowded_domain = crowded_domains[0]
                else:
                    PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, self.weights[domain_idx], self.best_obj, PBI_penalty) 
                                                        for sol_obj in last_level_by_domains[domain_idx]])
                    PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                    most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                    
                if len(last_level_by_domains[most_crowded_domain]) == 1:
                    worst_solution = locate_pareto_worst(self.pareto_levels.levels, self.weights, self.best_obj, PBI_penalty)                            
                else:
                    PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, self.weights, self.best_obj, PBI_penalty), 
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
                    worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        
        self.pareto_levels.delete_point(worst_solution)
        
        
    def optimize(self, neighborhood_selector, delta, neighborhood_selector_params, *kwargs):
        assert not type(self.best_obj) == type(None)
        cond = True
        while cond:
            for weight_idx in np.arange(len(self.weights)):
                parent_idxs = self.mating_selection(weight_idx, self.weights, self.neighborhood_lists, self.pareto_levels.population,
                                               neighborhood_selector, neighborhood_selector_params, delta)
                offsprings = self.evolutionary_operator.crossover([self.pareto_levels.population[idx] for idx in parent_idxs]) # В объекте эволюционного оператора выделять кроссовер
                try:    #exception for TypeError, when there is only one offspring, which is not placed in iterable object
                    for offspring_idx, offspring in enumerate(offsprings):
                        offsprings[offspring_idx] = self.evolutionary_operator.mutation(offspring)
                        self.update_population(offspring, kwargs['PBI_penalty'])
                except TypeError:
                    offsprings = self.evolutionary_operator.mutation(offsprings)
                    self.update_population(offsprings, kwargs['PBI_penalty'])
                    
                    
                    
class moeadd_optimizer_constrained(moeadd_optimizer):

    def set_constraints(self, *args) -> None:
        self.constrains = args


    def constaint_violation(self, solution_vals) -> float:
        return np.sum(np.fromiter(map(lambda constr: constr(solution_vals), self.constrains), dtype = float))

    def tournament_selection(self, candidate_1, candidate_2):
        if self.constaint_violation(candidate_1.x_vals) < self.constaint_violation(candidate_2.x_vals):
            return candidate_1
        elif self.constaint_violation(candidate_1.x_vals) > self.constaint_violation(candidate_2.x_vals):
            return candidate_2
        else:
            return np.random.choice((candidate_1, candidate_2))

    def update_population(self, offspring, PBI_penalty):
        self.pareto_levels.update(offspring)
        cv_values = np.zeros(len(self.pareto_levels.population))
        for sol_idx, solution in enumerate(self.pareto_levels.population):
            cv_val = self.constaint_violation(solution.x_vals)
            if cv_val > 0:
                cv_values[sol_idx] = cv_val 
        if sum(cv_values) == 0:
            if len(self.pareto_levels.levels) == 1:
                worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)
            else:
                if self.pareto_levels.levels[len(self.pareto_levels.levels) - 1] == 1:
                    domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
                    reference_solution = self.pareto_levels.levels[len(self.pareto_levels.levels) - 1][0]
                    reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
                    if len(domain_solutions[reference_solution_domain] == 1):
                        worst_solution = locate_pareto_worst(self.pareto_levels.levels, self.weights, self.best_obj, PBI_penalty)                            
                    else:
                        worst_solution = reference_solution
                else:
                    last_level_by_domains = population_to_sectors(self.pareto_levels.levels[len(self.pareto_levels.levels)-1], self.weights)
                    most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
                    crowded_domains = [domain_idx for domain_idx in np.arange(len(self.weights)) if len(last_level_by_domains[domain_idx]) == most_crowded_count]
    
                    if len(crowded_domains) == 1:
                        most_crowded_domain = crowded_domains[0]
                    else:
                        PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, self.weights[domain_idx], self.best_obj, PBI_penalty) 
                                                            for sol_obj in last_level_by_domains[domain_idx]])
                        PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                        most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                        
                    if len(last_level_by_domains[most_crowded_domain]) == 1:
                        worst_solution = locate_pareto_worst(self.pareto_levels, self.weights, self.best_obj, PBI_penalty)                            
                    else:
#                        print('the most crowded domain', most_crowded_domain)
                        PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, self.weights[most_crowded_domain], self.best_obj, PBI_penalty), 
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
#                        print('PBIS', PBIS, last_level_by_domains)
                        worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        else:
            infeasible = [solution for solution, _ in sorted(list(zip(self.pareto_levels.population, cv_values)), key = lambda pair: pair[1])]
            infeasible.reverse()
            print(np.nonzero(cv_values))
            infeasible = infeasible[:np.nonzero(cv_values)[0].size]
            deleted = False
            domain_solutions = population_to_sectors(self.pareto_levels.population, self.weights)
            
            for infeasable_element in infeasible:
                domain_idx = [domain_idx for domain_idx, domain in enumerate(domain_solutions) if infeasable_element in domain][0]
                if len(domain_solutions[domain_idx]) > 1:
                    deleted = True
                    worst_solution = infeasable_element
                    break
            if not deleted:
                worst_solution = infeasible[0]
        
        self.pareto_levels.delete_point(worst_solution)
#        if len(self.pareto_levels.population) != sum([len(level) for level in self.pareto_levels.levels]):
#            for level_idx, level in enumerate(self.pareto_levels.levels):
#                print(level_idx, [(solution.x_vals, solution.obj_fun) for solution in level])
#            print('\n', [solution.x_vals for solution in self.pareto_levels.population])
#            raise Exception('Something went wrong in point deletion, while adding ', offspring.x_vals, 'instead of ', worst_solution.x_vals)
            
    def optimize(self, neighborhood_selector, delta, neighborhood_selector_params, epochs, PBI_penalty):
        assert not type(self.best_obj) == type(None)
        for epoch_idx in np.arange(epochs):
            for weight_idx in np.arange(len(self.weights)):
                print(epoch_idx, weight_idx)
                parent_idxs = self.mating_selection(weight_idx, self.weights, self.neighborhood_lists, self.pareto_levels.population,
                                               neighborhood_selector, neighborhood_selector_params, delta)
                if len(parent_idxs) % 2:
                    parent_idxs = parent_idxs[:-1]
                np.random.shuffle(parent_idxs) 
                parents_selected = [self.tournament_selection(self.pareto_levels.population[int(parent_idxs[2*p_metaidx])], 
                                        self.pareto_levels.population[int(parent_idxs[2*p_metaidx+1])]) for 
                                        p_metaidx in np.arange(int(len(parent_idxs)/2.))]
                
                offsprings = self.evolutionary_operator.crossover(parents_selected) # В объекте эволюционного оператора выделять кроссовер
                for offspring_idx, offspring in enumerate(offsprings):
                    while True:
                        temp_offspring = self.evolutionary_operator.mutation(offspring)
                        if not np.any([temp_offspring == solution for solution in self.pareto_levels.population]):
                            break
                    self.update_population(temp_offspring, PBI_penalty)
