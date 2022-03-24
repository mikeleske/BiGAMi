from deap import base
from deap import creator
from deap import tools

import multiprocessing

import engine
import config
from model import regressors, classifiers

import numpy as np
import random
import os
from itertools import compress
import array
import collections
from collections import defaultdict
import pprint

from deap_custom import mutFlipOne

pp = pprint.PrettyPrinter(indent=4)

class GA:

    def __init__(self, GA_PARAMS, data, path, search_id):
        self.GA_PARAMS = GA_PARAMS
        self.pop_size = GA_PARAMS['pop_size']
        self.max_iter = GA_PARAMS['max_iter']
        self.penalty_denominator = GA_PARAMS['penalty_denominator']
        self.data = data
        self.seen_individuals = defaultdict(int)
        self.plot_data = collections.defaultdict(list)

        self.toolbox = self.__build_toolbox()

        self.path = '{}/{}'.format(path, search_id)
        self.search_id = search_id

        self.history = {}
        self.history_kfold = []

        self.stats = {
            'kfold': [],
            'test': [],
        }

    def __build_toolbox(self):

        print('\nBuilding GA toolbox.')

        try:
            del creator.FitnessMax
            del creator.Individual
        except:
            pass

        creator.create("FitnessMax", base.Fitness, weights=self.GA_PARAMS['weights'])
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Attribute generator
        if self.GA_PARAMS['init'] == 'zero':
            toolbox.register("attr_bool",  random.randint, 0, 0)
        elif self.GA_PARAMS['init'] == 'random':
            toolbox.register("attr_bool", random.randint, 0, 1)
        elif self.GA_PARAMS['init'] == 'full':
            toolbox.register("attr_bool",  random.randint, 1, 1)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.GA_PARAMS['gen_size'])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.eval, flag='kfold')
        toolbox.register("evaluate_best_kfold", self.eval, flag='best-kfold')
        toolbox.register("evaluate_best_test", self.eval, flag='best-test')

        if self.GA_PARAMS['crossover'] == '1p':
            toolbox.register("mate", tools.cxOnePoint)
        if self.GA_PARAMS['crossover'] == '2p':
            toolbox.register("mate", tools.cxTwoPoint)
        if self.GA_PARAMS['crossover'] == 'uniform':
            toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate_init", tools.mutFlipBit, indpb=self.GA_PARAMS['mutate_init_prob'])
        toolbox.register("mutate", mutFlipOne)
        toolbox.register("mutate_dup", mutFlipOne)
        toolbox.register("select", tools.selTournament, tournsize=3)

        toolbox.register("best1", tools.selBest, k=1, fit_attr='fitness')
        toolbox.register("bestN", tools.selBest, k=self.GA_PARAMS['bestN'])

        # Process Pool of 4 workers
        #pool = multiprocessing.Pool(processes=8)
        #toolbox.register("map", pool.map)

        return toolbox


    def search(self):

        #os.mkdir(self.path)

        print('\nGenerating initial population.')
        pop = self.toolbox.population(n=self.pop_size)

        for ind in pop:
            self.toolbox.mutate_init(ind)

        self.check_duplicate_ind(pop)

        # Evaluate the entire population
        print('Evaluating initial population.')
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # CXPB  is the probability with which two individuals are crossed
        # MUTPB is the probability for mutating an individual
        CXPB  = self.GA_PARAMS['CXPB']
        MUTPB = self.GA_PARAMS['MUTPB']

        # Begin the evolution
        for i in range(self.max_iter):
            print('\nGeneration {}.'.format(i + 1))

            self.track_cv_all = []

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop) - self.GA_PARAMS['bestN'])
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            self.check_duplicate_ind(offspring)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            for elite in self.toolbox.bestN(pop):
                offspring.append(elite)

            pop[:] = offspring

            # Get the fittest individual
            best = self.toolbox.best1(offspring)
            fitness_best_kfold = list(map(self.toolbox.evaluate_best_kfold, best))
            fitness_best_test = list(map(self.toolbox.evaluate_best_test, best))

            self.update_stats(best, fitness_best_kfold, fitness_best_test)

        return pop, self.toolbox.best1(pop)

    def get_population_diversity(self, pop):
        # Get population diversity
        d = []
        for ind in pop:
            d = d + list(compress(range(len(ind)), ind))

        return len(np.unique(d))

    def update_stats(self, best, fitness_best_kfold, fitness_best_test):
        metric_kfold = fitness_best_kfold[0][0]
        metric_test  = fitness_best_test[0][0]
        best_gensize = sum(best[0])

        mask = list(compress(range(len(best[0])), best[0]))
        features = list(self.data.features[mask])

        self.stats['kfold'].append((
            self.search_id,
            best_gensize,
            metric_kfold,
            np.round(fitness_best_kfold[0][0] - fitness_best_kfold[0][1], 3),
            mask,
            features))

        self.stats['test'].append((
            self.search_id,
            best_gensize,
            metric_test,
            np.round(fitness_best_test[0][0] - fitness_best_test[0][1], 3),
            mask,
            features))

    def hash_ind(self, ind):
        s = ''.join(map(str, ind.tolist()))
        return hash(s)

    def check_duplicate_ind(self, pop):
        for ind in pop:
            hs = self.hash_ind(ind)

            if not self.seen_individuals[hs]:
                self.seen_individuals[hs] = 1
            else:
                # Handle duplicate individuals
                while self.seen_individuals[hs]:
                    self.toolbox.mutate_dup(ind)
                    hs = self.hash_ind(ind)
                del ind.fitness.values
                self.seen_individuals[hs] = 1


    def get_model(self, task_type, alg):
        if task_type == 'regression':
            return regressors[alg]
        elif task_type == 'classification':
            return classifiers[alg]

    def eval(self, individual, flag):

        test_fold       = config.DATA_PARAMS['folds'] - 1
        multipe_samples = config.DATA_PARAMS['task']['multiple_samples']

        metric    = config.TRAIN_PARAMS['metric']
        task_type = config.TRAIN_PARAMS['type']
        alg       = config.TRAIN_PARAMS['alg']

        mask = list(compress(range(len(individual)), individual))

        if len(mask)==0:
            return (-9999999999, 0)

        scores = []
        sum_feature_importances = defaultdict(float)

        if flag == 'kfold' or flag == 'best-kfold':
            for fold in range(config.DATA_PARAMS['folds'] - 1):

                model = self.get_model(task_type, alg)
                score = engine.train(self.data.df, multipe_samples, model, metric, fold, test_fold, mask)
                scores.append(score)

                try:    fi = sorted(zip(model.feature_importances_, self.data.features[mask]))
                except: pass

                try:    fi = sorted(zip(model.coef_[0], self.data.features[mask]))
                except: pass

                for value, feature in fi:
                    sum_feature_importances[feature] += value

        if flag == 'best-test':
            fold = test_fold
            model = self.get_model(task_type, alg)
            score = engine.train(self.data.df, multipe_samples, model, metric, fold, test_fold, mask)
            scores.append(score)

        if flag == 'kfold':
            # Remove genes with importance 0.0
            for key, value in sum_feature_importances.items():
                if np.abs(value) < 0.5:
                    feat_idx = np.where(self.data.features==key)[0][0]
                    individual[feat_idx] = 0

        if flag == 'best-kfold':
            print('  eval_best_kfold_fn: Best --> Avg Score={} ({}), Max Score={}, features={}, mo_fitness={}'.format(
                    np.round(np.mean(scores)+np.min(scores), 3),
                    np.round(np.std(scores), 3),
                    np.round(np.max(scores), 3),
                    len(mask),
                    np.round(np.mean(scores) - len(mask)/self.penalty_denominator, 3)
                )
            )

        if flag == 'best-test':
            print('  eval_best_test_fn:  Best --> Avg Score={} ({}), Max Score={}, features={}, mo_fitness={}'.format(
                    np.round(np.mean(scores), 3),
                    np.round(np.std(scores), 3),
                    np.round(np.max(scores), 3),
                    len(mask),
                    np.round(np.mean(scores) - len(mask)/self.penalty_denominator, 3)
                )
            )

        if flag == 'best-test':
            return np.round(np.mean(scores), 3), len(mask)/self.penalty_denominator
        else:
            return np.round(np.mean(scores)+np.min(scores), 3), len(mask)/self.penalty_denominator
