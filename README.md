# Genetic Algorithm

## Brief Descriptions
A simple version of genetic algorithm python package, working with pandas.DataFrame structure, which followed elite strategy and bit string mutation, with multiple fitness type can be chosen.  

## Introduction of Genetic Algorithm
> Genetic Algorithm (GA) is a search-based optimization technique based on the principles of Genetics and Natural Selection. It is frequently used to find optimal or near-optimal solutions to difficult problems which otherwise would take a lifetime to solve. It is frequently used to solve optimization problems, in research, and in machine learning. More datails in : https://medium.com/xrpractices/reinforcement-learning-vs-genetic-algorithm-ai-for-simulations-f1f484969c56


## Use Guide
### Import Package
Download the python script file and save it in the same folder with your code.  
```
from genetic_algorithm import GA
```

### Define Model
```
GAModel = GA(train_x, train_y, test_x, test_y, mutuate_rate=0.15, cross_rate=0.99, mode="A")
```
***mutuate_rate*** is the probability that the bit inside a structure mutuate, which is preset at 0.15.  
***cross_rate*** is the probability that a crossover happen, which is preset at 0.99.  
***mode*** is the mode for calculating fittness, can be chosen from {"A" : Accuracy, "P" : Precision, "R" : Recall, "F" : F1}, preset at "A".  

### Initialize Population
```
GAModel.initialize(population_size=100)
```
***population_size*** is the size for the orginal population, which is preset at 100.  

### Start Training
```
GAModel.generate(loop=10, early_stop=5)
```
***loop*** is the number of generations sholud be trained, which is preset at 10.  
***early_stop*** is how many generations that the fitness value not changing stop training, which is preset at 5 and can be shut down by setting False.  

### Save and Load Model
```
GAModel.save_pickle(path="yourFolder/")
GAModel.load_pickle(path="yourFolder/")
```
***path*** is the folder you want to save or load your model, the model will be save as three file as shown in picture below, fitlist.pkl save the of highest fitness of every generation as a list, population.pkl save the structure and fitness of every model in the population as a dictionary, record.pkl save the structure and fitness of every model of all historical model as a dictionary.  

![](https://i.imgur.com/FqS1sst.png)

### Draw Training History
You can also draw the training history with the help of other package, here is a simple demonstration.  
```
import matplotlib.pyplot as plt
fit_list = GAModel.fit_list
plt.plot(fit_list)
plt.title("Genetic Algorithm")
plt.xlabel("generation")
plt.ylabel("fitness")
plt.show()
```
![](https://i.imgur.com/0bJArle.png)
