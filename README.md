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

## Objective
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
python describe.py [-h] [-f] [-c] dataset_file
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


## Objective of the graph functions
Plot some relevant graphs of Poudlard's student train set: histogram, pair plot and scatter plot. They should answer the foloowing questions:
- Which subject has the most homogeneous grades across the different houses?
- What are the 2 features that are similar?
- From the pair plot what are the features you will use to train the logistic regression?

### How to run the program
You must have python3 installed.  
First set up the virtual environment: ```source set_up/set_env.sh```

To run the different graphs do:
<pre>
python histogram.py [-h] [-a] dataset_file
python pair_plot.py [-h] dataset_file
python scatter_plot.py [-h] [-d] dataset_file
</pre>

The only required input is the input filename (`dataset_file`). It must be a csv. The rest is optional.

### Options
<pre>
Options for histogram
  -a, --all     show all histograms.

Options for scatter_plot
  -d, --detailed  show all scatter plots on different pages with more details.
</pre>  

### Results displayed
The terminal indicates if a pdf or png file has been created that compiles the graph.



## Objective of the logreg_train function
This function creates a weights.csv file that compiles the weights resulting of the training of the thetas (weights) through
logistic regression algorithm for each of Poudlard's house.

These weights are used on new input the give a predicted probability of this new input to belong to a given house.


### How to run the program
You must have python3 installed.  
First set up the virtual environment: ```source set_up/set_env.sh```

To run the different graphs do:
<pre>
python logreg_train.py [-h] [-l LEARNING_RATE] [-e EPSILON] [-r REG_PARAM]
                       [-s SPLIT] [-v] [-a {simple,full}]
                       [-m {batch,mini_batch,stochastic}] [-b BATCH_SIZE]
                       [-i ITERATIONS]
                       dataset_train_file
</pre>

The only required input is the input filename (`dataset_file`). It must be a csv. The rest is optional.
By default, options are set to use a batch gradient descent with optimized hyperparameters.

### Options
<pre>
  -h, --help            show this help message and exit
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Choose the learning rate.
  -e EPSILON, --epsilon EPSILON
                        Choose the epsilon when iterations should stop.
  -r REG_PARAM, --reg_param REG_PARAM
                        Choose the regularization parameter.
  -s SPLIT, --split SPLIT
                        Choose the train_size split.
  -v, --visualisation   Display cost function.
  -a {simple,full}, --accuracy {simple,full}
                        Display train accuracy if simple and more information
                        if full
  -m {batch,mini_batch,stochastic}, --mode {batch,mini_batch,stochastic}
                        Choose gradient descent mode.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        For mini_batch. Pick the mini_batch size.
  -i ITERATIONS, --iterations ITERATIONS
                        For mini_batch and stochastic. Pick the number of
                        iterations.
</pre>  

### Results displayed
By default no results are displayed but a weights.csv file is created in the folder.
If -v option is set, it also shows the graph with the evolution of the cost functions (one per house) per iteration.

If -a option is set to full. A few information on the training stats are displayed:
<pre>
Train total accuracy is: 0.9833333333333333

Accuracy per house is:
          Gryffindor  Hufflepuff  Ravenclaw  Slytherin
accuracy         1.0    0.982456   0.985185   0.964706

Wrong predictions are:
           truth  prediction
184    Ravenclaw   Slytherin
212    Slytherin   Ravenclaw
255    Slytherin   Ravenclaw
456    Slytherin   Ravenclaw
944   Hufflepuff   Ravenclaw
1143  Hufflepuff   Ravenclaw
1191  Hufflepuff  Gryffindor
1255   Ravenclaw  Hufflepuff
Train total accuracy is: 0.9833333333333333

</pre>

If -a option is set to simple. Only the train total accuracy is displayed.



## Objective of the logreg_predict function
This function uses a dataset file and the weights to generate a houses.csv file with the predicted house for each student.


### How to run the program
You must have python3 installed.  
First set up the virtual environment: ```source set_up/set_env.sh```

To run the different graphs do:
<pre>
python  logreg_predict.py [-h] dataset_test_file weights
</pre>

The only required input is the input filename (`dataset_file`). It must be a csv.


### Results displayed
No results are displayed but a houses.csv file is created in the folder with the prediction related to the dataset_test files.

## Unit tests
Running ```python tests.py``` will run the unit tests written for the program. 
Those tests are mainly for describe.py. They are also linked to travis-ci.
