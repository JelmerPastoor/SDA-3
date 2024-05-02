It is very important to have the file 'llh_values.npy' IN THE SAME DIRECTORY AS MAIN.PY, otherwise the program will result in an error when calculating the error on the two simultaneous fitted parameters 
(the additional exercise number 2). 
The python module Warning is imported to silence the warning that the p_value of the AD test is lower than 0.001, which (without a method given to the function) returns a warning. 

The use of the definition 'loglikelihood_2params' is commented out, since (for the amount of values it has to check for sigma and fnuc) it takes >10 minutes to run. The best values can be found in the variables
	best_likelihood_2fit
	best_sigma_2fit
	best_fnuc_2fit
	llh_values

