# Genetic Algorithm
# Author: Meng-Chieh, Liu  
# Date: 2022/5/25


import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from operator import itemgetter
import pickle
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def hash(structure):
    return "".join(str(x) for x in structure)


def unhash(string_structure):
    return [int(x) for x in string_structure]


def switch(s):
    if s == 0:
        return 1
    return 0


class GA():  
    # Initialize class with data, rates and mode
    def __init__(self, train_x, train_y, test_x, test_y, mutate_rate=0.01, cross_rate=1, mode="A"):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.mutate_rate = mutate_rate
        self.cross_rate = cross_rate
        self.record = {}
        self.population = {}
        self.fit_list = []
        self.mode = mode


    # Calculate fitness according to accuracy/precision/recall/f1
    def fitness(self, structure):
        col_list = []
        for col, s in zip(self.train_x.columns, structure):
            if s ==1:
                col_list.append(col)
        new_train_x = self.train_x[col_list]
        new_test_x = self.test_x[col_list]
        clf = tree.DecisionTreeClassifier(random_state= 42)
        clf_model = clf.fit(new_train_x, self.train_y)
        pred_y = clf_model.predict(new_test_x)
        if self.mode == "A":
            return accuracy_score(self.test_y, pred_y)
        if self.mode == "P":
            return precision_score(self.test_y, pred_y)
        if self.mode == "R":
            return recall_score(self.test_y, pred_y)
        if self.mode == "F":
            return f1_score(self.test_y, pred_y)


    # Initialize population
    def initialize(self, population_size=100):
        print("Initializing......")
        dimension = len(self.train_x.columns)
        for i in tqdm(range(population_size)):
            while True:
                structure = []
                for d in range(dimension):
                    structure.append(round(random.random()))
                if sum(structure) == 0:
                    continue
                try:
                    self.population[hash(structure)] = self.record[hash(structure)]
                except:
                    fit = GA.fitness(self, structure)
                    self.population[hash(structure)] = fit
                    self.record[hash(structure)] = fit
                    break
        self.population = dict(sorted(self.population.items(), key = itemgetter(1), reverse = True))
        self.fit_list = [list(self.population.values())[0]]


    # mutate followed bit string mutation
    def mutate(self, structure):
        new_structure = structure
        for i, s in enumerate(structure):
            if random.random() < self.mutate_rate:
                new_structure[i] = switch(s)
        return new_structure


    # get highest fitness value of all time
    def get_best_fitness(self):
        return list(self.population.values())[0]


    # Choose parents followed elite strategy
    def crossing(self):
        keys, weights = zip(*self.population.items())
        probs = np.array(weights, dtype=float) / float(sum(weights))
        N = len(keys)
        d = len(keys[0])
        split_point = int(d/2)
        for i in range(N):
            key1, key2 = np.random.choice(keys, 2, p=probs).tolist()
            if random.random() < self.cross_rate:
                new_structure = unhash(key1)[:split_point] + unhash(key2)[split_point:]
                new_structure = GA.mutate(self, new_structure)
                try:
                    self.population[hash(new_structure)] = self.record[hash(new_structure)]
                except:
                    fit = GA.fitness(self, new_structure)
                    self.population[hash(new_structure)] = fit
                    self.record[hash(new_structure)] = fit
        self.population = dict(sorted(self.population.items(), key = itemgetter(1), reverse = True)[:N])


    # Generate offsprings, run untill meeting criteria
    def generate(self, loop = 50, early_stop = 5):
        print("Generating......")
        stop_signal = 0
        prev_best_fit = list(self.population.values())[0]
        for i in tqdm(range(loop)):
            GA.crossing(self)
            best_fit = GA.get_best_fitness(self)
            self.fit_list.append(best_fit)
            if early_stop != False:
                if best_fit == prev_best_fit:
                    stop_signal +=1
                else:
                    stop_signal = 0
                if stop_signal == early_stop: 
                    break
            prev_best_fit = best_fit


    # get average columns result of n structures with highest fitness
    def get_col(self, n=5):
        df = pd.DataFrame()
        col_list = self.train_x.columns
        for i, fit in enumerate(list(self.population.keys())[:n]):
            df[i] = unhash(fit)
        structure = round(df.sum(axis = 1)/n)
        col_list = []
        for col, s in zip(self.train_x.columns, structure):
            if s ==1:
                col_list.append(col)
        return col_list


    # save training result as pickle
    def save_pickle(self, path=""):
        with open(path+'record.pkl', 'wb') as fp:
            pickle.dump(self.record, fp)
        with open(path+'population.pkl', 'wb') as fp:
            pickle.dump(self.population, fp)
        with open(path+'fit_list.pkl', 'wb') as fp:
            pickle.dump(self.fit_list, fp)


    # load pretrained model from pickle
    def load_pickle(self, path=""):
        with open(path+'record.pkl', 'rb') as fp:
            self.record = pickle.load(fp)
        with open(path+'population.pkl', 'rb') as fp:
            self.population = pickle.load(fp)
        with open(path+'fit_list.pkl', 'rb') as fp:
            self.fit_list = pickle.load(fp)
