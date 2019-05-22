[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Build Status](https://travis-ci.com/blopax/dslr.svg?branch=master)](https://travis-ci.com/blopax/dslr)

# DSLR

DSLR in python 3.7. 

It is a 42 project aiming to code a multiclass classifier from scratch by creating:
- a describe function (similar to pandas method) for numerical columns of a dataset.
- a few scripts that renders a few useful graphs to analyze the data set.
- a one-vs-all algorithm using logistic classifier from scratch, implementing a gradient descent.

A set of options have been added to the project required to:
- be able to choose the gradient descent mode (batch, stochastic or mini-batch) with relevant hyper-parameters (batch-size, iterations)
- be able to get more information on the dataset that with the describe function fo pandas
- be able to define various hyper parameters: learning rate alpha, regularization parameter, cross-validation split
- for batch gradient descent define the epsilon from which the gradient descent stops iterating.
- show the global and per class accuracy on the cross-validation test set and all the mis-predictions.
- show the plot of the cost functions given the iterations for the 4 class 

I used pandas and numpy but no functions that would do the ML algorithms or the describe function for me.
The aim was to not use a ready-made library (like sklearn) but to understand how the algos work.

## Objective of the describe function
Given a training data set of Poudlards students, build a classifier that predicts to which house students of an unknown dataset belong with an accuracy of 98%.
To do so, analyze first the data using a describe function that must be coded from scratch (similar to the pandas method) 
, plot some relevant graphs given some constraints (histogram, pair plot and scatter plot) and then build a train and predic script.


## Objective of the describe function
Given a training data set of Poudlard's students, analyze first the numerical data using a describe function that must be coded from scratch (similar to the pandas one). 

### How to run the program
You must have python3 installed.  
First set up the virtual environment: ```source set_up/set_env.sh```

To run the describe program do:
<pre>
describe.py [-h] [-f] [-c] dataset_file
</pre>

The only required input is the input filename (`dataset_file`). It must be a csv. The rest is optional.

### Options
<pre>
  -h, --help     show this help message and exit
  -f, --full     add more info to description.
  -c, --compare  show the pandas describe method to compare results.
</pre>  

### Results displayed
Basic results are of this type.
<pre>
          Arithmancy    Astronomy    Herbology  Defense Against the Dark Arts   Divination  ...  Transfiguration      Potions  Care of Magical Creatures       Charms       Flying
Count    1566.000000  1568.000000  1567.000000                    1569.000000  1561.000000  ...      1566.000000  1570.000000                1560.000000  1600.000000  1600.000000
Mean    49634.570243    39.797131     1.141020                      -0.387863     3.153910  ...      1030.096946     5.950373                  -0.053427  -243.374409    21.958012
Std     16679.806036   520.298268     5.219682                       5.212794     4.155301  ...        44.125116     3.147854                   0.971457     8.783640    97.631602
Min    -24370.000000  -966.740546   -10.295663                     -10.162119    -8.727000  ...       906.627320    -4.697484                  -3.313676  -261.048920  -181.470000
25%     38511.500000  -489.551387    -4.308182                      -5.259095     3.099000  ...      1026.209993     3.646785                  -0.671606  -250.652600   -41.870000
50%     49013.500000   260.289446     3.469012                      -2.589342     4.624000  ...      1045.506996     5.874837                  -0.044811  -244.867765    -2.515000
75%     60811.250000   524.771949     5.419183                       4.904680     5.667000  ...      1058.436410     8.248173                   0.589919  -232.552305    50.560000
Max    104956.000000  1016.211940    11.612895                       9.667405    10.032000  ...      1098.958201    13.536762                   3.056546  -225.428140   279.070000
</pre>

If -f option: it adds the first and last centile and decile.
If -c option: it displays below the pandas describe() method result


## Objective of the describe function
Plot some relevant graphs of Poudlard's student train set: histogram, pair plot and scatter plot. 

### How to run the program
You must have python3 installed.  
First set up the virtual environment: ```source set_up/set_env.sh```

To run the describe program do:
<pre>
describe.py [-h] [-f] [-c] dataset_file
</pre>

The only required input is the input filename (`dataset_file`). It must be a csv. The rest is optional.

### Options
<pre>
  -h, --help     show this help message and exit
  -f, --full     add more info to description.
  -c, --compare  show the pandas describe method to compare results.
</pre>  

### Results displayed
Basic results are of this type.
<pre>
          Arithmancy    Astronomy    Herbology  Defense Against the Dark Arts   Divination  ...  Transfiguration      Potions  Care of Magical Creatures       Charms       Flying
Count    1566.000000  1568.000000  1567.000000                    1569.000000  1561.000000  ...      1566.000000  1570.000000                1560.000000  1600.000000  1600.000000
Mean    49634.570243    39.797131     1.141020                      -0.387863     3.153910  ...      1030.096946     5.950373                  -0.053427  -243.374409    21.958012
Std     16679.806036   520.298268     5.219682                       5.212794     4.155301  ...        44.125116     3.147854                   0.971457     8.783640    97.631602
Min    -24370.000000  -966.740546   -10.295663                     -10.162119    -8.727000  ...       906.627320    -4.697484                  -3.313676  -261.048920  -181.470000
25%     38511.500000  -489.551387    -4.308182                      -5.259095     3.099000  ...      1026.209993     3.646785                  -0.671606  -250.652600   -41.870000
50%     49013.500000   260.289446     3.469012                      -2.589342     4.624000  ...      1045.506996     5.874837                  -0.044811  -244.867765    -2.515000
75%     60811.250000   524.771949     5.419183                       4.904680     5.667000  ...      1058.436410     8.248173                   0.589919  -232.552305    50.560000
Max    104956.000000  1016.211940    11.612895                       9.667405    10.032000  ...      1098.958201    13.536762                   3.056546  -225.428140   279.070000
</pre>

If -f option: it adds the first and last centile and decile.
If -c option: it displays below the pandas describe() method result


## Unit tests
Running ```python tests.py``` will run the unit tests written for the program. 
Those tests are mainly for describe.py. They are also linked to travis-ci.
