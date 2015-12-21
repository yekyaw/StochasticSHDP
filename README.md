# Generalized Supervised HDP
This implements online variational inference for Generalized Supervised HDP.

## Requirements
Python 3.4, Numpy, Scipy

## Execution
Run python run_online_hdp.py with the desired parameters. Run with
--help for a list of available parameters.

## Required parameters
--D The number of documents in the data file. 

--W The number of distinct words in the vocabulary.

--max_iter The maximum number of iterations to run. Setting this to
(D*2/batchsize) seems to work well.

--directory The root output directory.

--corpus_name  The name of the corpus. The output will be saved under
directory/corpus_name.

--data_path The training data path. See below for the format of the
file.

--responses The responses file path. See below for the format of the file.

## Optional parameters
--T The truncation level for the top level Dirichlet sticks.

--K The truncation level for the second level Dirichlet sticks.

--eta The topic Dirichlet

--gamma The top-level Dirichlet

--alpha The second level Dirichlet

--kappa The learning rate for stochastic variational inference

--tau The delay for stochastic variational inference

--y_scale This is the multiplier for the response term, which can be
adjusted to change the influence of the response terms.

--penalty_lambda The lambda value for the penalty terms.

--l1_ratio The penalty will be set to (l1_ratio * L1 term+(1 -
  l1_ratio) * L2 term)

--batchsize The batch size

--max_time Max time to run training in seconds

--var_converge Used to test for convergence. We assume convergence if
the relative change in the ELBO is less than var_converge.

--test_data_path The test data path. Should be the same format as the
training data.

--save_lag Lag for saving the model.

--scale Scale for the learning rate.

--adding_noise Adds noise to the first few iterations.

## Data file format
Documents should be separated by newline characters. Within each
document, the delimiter should be spaces. Each document should first
specify the number of distinct words in the document. Next, it should
specify the label values, which should be delimited by
commas. Finally, it should specify the words in a bag of words format,
i.e. WORD:FREQUENCY.

### Example:
3 0,2 0:5 1:3 2:1

This specifies that there are 3 distinct terms in this document and 2
label values (0 and 2). There are 5 occurences of word 0, 3
occurrences of word 1, and 1 occurrence of word 2.

## Response file format
There should be a line specifying the response type for each
response. For example, if each document has two label values, there
should be two entries in the file. The model currently supports 3
response types: Poisson, Categorical, and Bernoulli. If the response
type is Categorical, it should also include the number of possible
outcomes (with a colon delimiter).

### Example
Poisson

Categorical:3

Bernoulli

This specifies that the first response is a Poisson, the second
response is a Categorical with 3 possible outcomes, and the third
response is a Bernoulli.

## Acknowledgements
Large chunks of the code were adapted from Chong Wang's online HDP
code.

Chong Wang, John Paisley and David M. Blei. Online variational
inference for the hierarchical Dirichlet process. In
AISTATS 2011. Oral presentation. [PDF](http://www.cs.princeton.edu/~chongw/papers/WangPaisleyBlei2011.pdf)
