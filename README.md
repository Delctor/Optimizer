# tpe
I have been interested in function optimization since some months so i decided to made a tree-structured parzen estimator
in python just using numpy, this TPE divide the tested parameters in "good" and "bad" set of parameters with a gamma 
then use a PSO(Particle Swarm Optimization) algorithm to get new parameters to test in a function ("ei") that calculates 
the density of each set of parameters suggested by the PSO in the "good" and "bad "set of parameters, 
after this the pso returns the best set of parameters to test in the objective function. 
The "ei" function also filters previously tested parameters so that the algorithm does not get stuck in a single set of parameters.
