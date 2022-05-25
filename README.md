# Genetic Algorithm

## Descriptions
A simple version of genetic algorithm python package, which followed elite strategy and bit string mutation, with multiple fitness type can be chosen.  

## Guide
### Initialize Model
***mutuate_rate*** is the probability that the bit inside a structure mutuate, which is preset at 0.15.  
***cross_rate*** is the probability that a crossover happen, which is preset at 0.99.  
***mode*** is the mode for calculating fittness, can be chosen from {"A" : Accuracy, "P" : Precision, "R" : Recall, "F" : F1}, preset at "A".  
```
GAModel = GA(train_x, train_y, test_x, test_y, mutuate_rate=0.15, cross_rate=0.99, mode="A")
```

### Initialize Population
***population_size*** is the size for the orginal population, which is preset at 100.  
```
GAModel.initialize(population_size=100)
```

### Start Training
***loop*** is the number of generations sholud be trained, which is preset at 10.  
***earlystop*** is how many generations that the fitness value not changing stop training, which is preset at 5 and can be shut down by setting False.  
```
GAModel.generate(loop=10, early_stop=5)
```
