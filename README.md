# Generalized Supervised HDP
This implements online variational inference for Generalized Supervised HDP.

## Requirements
Python 3.4, Numpy, Scipy

## Execution
Run python run_online_hdp.py with the desired parameters. Run with --help for a list
of available parameters.

## Required parameters
--D The number of documents in the data file. Documents should be delimited by "\n".
--W The number of distinct words in the vocabulary.
--max_iter The maximum number of iterations to run. Setting this to (D*2/batchsize) seems to work well.
--directory The root output directory.
--corpus_name  The name of the corpus. The output will be saved under directory/corpus_name.
--data_path The training data path. See below for the format of the file.
--responses The responses file path. See below for the format of the file.

## Optional parameters
--T The truncation level for the top level Dirichlet sticks.
--K The truncation level for the second level Dirichlet sticks.
--eta The topic Dirichlet
--gamma The top-level Dirichlet
--alpha The second level Dirichlet
--kappa The learning rate for stochastic variational inference
--tau The delay for stochastic variational inference
--y_scale This is the multiplier for the response term, which can be adjusted to change the influence of the response terms.
--penalty_lambda The lambda value for the penalty terms.
--l1_ratio The penalty will be set to (l1_ratio*L1 term+(1-l1_ration)*L2 term)
--batch_size The batch size
--max_time Max time to run training in seconds
--var_converge Used to test for convergence. We assume convergence if the relative change in the ELBO is less than var_converge.
--test_data_path The test data path. Should be the same format as the training data.
--save_lag Lag for saving the model.
--scale Scale for the learning rate.
--adding_noise Adds noise to the first few iterations.

## Acknowledgements
Large chunks of the code were adapted from Chong Wang's online HDP code.
Chong Wang, John Paisley and David M. Blei. Online variational inference for the hierarchical Dirichlet process. In AISTATS 2011. Oral presentation. [PDF](http://www.cs.princeton.edu/~chongw/papers/WangPaisleyBlei2011.pdf)
