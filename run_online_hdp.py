import os
from corpus import *
import onlinehdp
import pickle
import random, time
from optparse import OptionParser
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
np = onlinehdp.np

def parse_args():
  parser = OptionParser()
  parser.set_defaults(C=None, T=100, K=10, D=-1, W=-1, eta=0.01, alpha=1.0, gamma=1.0,
                      kappa=0.9, tau=1., batchsize=500, max_time=-1,
                      max_iter=-1, var_converge=0.0001, 
                      corpus_name=None, data_path=None, test_data_path=None, 
                      test_data_path_in_folds=None, directory=None, save_lag=500, pass_ratio=0.5,
                      new_init=False, scale=1.0, adding_noise=False,
                      seq_mode=False, fixed_lag=False)

  parser.add_option("--responses", type="string", dest="responses",
                    help="response types [None]")
  parser.add_option("--T", type="int", dest="T",
                    help="top level truncation [100]")
  parser.add_option("--K", type="int", dest="K",
                    help="second level truncation [10]")
  parser.add_option("--D", type="int", dest="D",
                    help="number of documents [-1]")
  parser.add_option("--W", type="int", dest="W",
                    help="size of vocabulary [-1]")
  parser.add_option("--eta", type="float", dest="eta",
                    help="the topic Dirichlet [0.01]")
  parser.add_option("--alpha", type="float", dest="alpha",
                    help="alpha value [1.0]")
  parser.add_option("--gamma", type="float", dest="gamma",
                    help="gamma value [1.0]")
  parser.add_option("--kappa", type="float", dest="kappa",
                    help="learning rate [0.9]")
  parser.add_option("--tau", type="float", dest="tau",
                    help="slow down [1.0]")
  parser.add_option("--batchsize", type="int", dest="batchsize",
                    help="batch size [500]")
  parser.add_option("--max_time", type="int", dest="max_time",
                    help="max time to run training in seconds [100]")
  parser.add_option("--max_iter", type="int", dest="max_iter",
                    help="max iteration to run training [-1]")
  parser.add_option("--var_converge", type="float", dest="var_converge",
                    help="relative change on doc lower bound [0.0001]")
  parser.add_option("--corpus_name", type="string", dest="corpus_name",
                    help="the corpus name: nature, nyt or wiki [None]")
  parser.add_option("--data_path", type="string", dest="data_path",
                    help="training data path or pattern [None]")
  parser.add_option("--test_data_path", type="string", dest="test_data_path",
                    help="testing data path [None]")
  parser.add_option("--test_data_path_in_folds", type="string",
                    dest="test_data_path_in_folds",
                    help="testing data prefix for different folds [None], not used anymore")
  parser.add_option("--directory", type="string", dest="directory",
                    help="output directory [None]")
  parser.add_option("--save_lag", type="int", dest="save_lag",
                    help="the minimal saving lag, increasing as save_lag * 2^i, with max i as 10; default 500.")
  parser.add_option("--pass_ratio", type="float", dest="pass_ratio",
                    help="The pass ratio for each split of training data [0.5]")
  parser.add_option("--new_init", action="store_true", dest="new_init",
                    help="use new init or not")
  parser.add_option("--scale", type="float", dest="scale",
                    help="scaling parameter for learning rate [1.0]")
  parser.add_option("--adding_noise", action="store_true", dest="adding_noise",
                    help="adding noise to the first couple of iterations or not")
  parser.add_option("--seq_mode", action="store_true", dest="seq_mode",
                    help="processing the data in the sequential mode")
  parser.add_option("--fixed_lag", action="store_true", dest="fixed_lag",
                    help="fixing a saving lag")
  
  (options, args) = parser.parse_args()
  return options 

def run_online_hdp():
  # Command line options.
  options = parse_args()

  # Set the random seed.
  if options.seq_mode:
    train_file = open(options.data_path)
  else:
    train_filenames = glob(options.data_path)
    train_filenames.sort()
    num_train_splits = len(train_filenames)
    # This is used to determine when we reload some another split.
    num_of_doc_each_split = options.D/num_train_splits 
    # Pick a random split to start
    # cur_chosen_split = int(random.random() * num_train_splits)
    cur_chosen_split = 0 # deterministic choice
    cur_train_filename = train_filenames[cur_chosen_split]
    c_train = read_data(cur_train_filename)
  
  if options.test_data_path is not None:
    test_data_path = options.test_data_path
    c_test = read_data(test_data_path)

  if options.test_data_path_in_folds is not None:
    test_data_path_in_folds = options.test_data_path_in_folds
    test_data_in_folds_filenames = glob(test_data_path_in_folds)
    test_data_in_folds_filenames.sort()
    num_folds = len(test_data_in_folds_filenames)/2
    test_data_train_filenames = []
    test_data_test_filenames = []

    for i in range(num_folds):
      test_data_train_filenames.append(test_data_in_folds_filenames[2*i+1])
      test_data_test_filenames.append(test_data_in_folds_filenames[2*i])

  result_directory = "%s/corpus-%s-kappa-%.1f-tau-%.f-batchsize-%d" % (options.directory,
                                                                       options.corpus_name,
                                                                       options.kappa, 
                                                                       options.tau, 
                                                                       options.batchsize)
  print("creating directory %s" % result_directory)
  if not os.path.isdir(result_directory):
    os.makedirs(result_directory)

  options_file = open("%s/options.dat" % result_directory, "w")
  for opt, value in options.__dict__.items():
    options_file.write(str(opt) + " " + str(value) + "\n")
  options_file.close()

  print("creating online hdp instance.")
  responses = open(options.responses).read().splitlines()
  ohdp = onlinehdp.online_hdp(responses, options.T, options.K, options.D, options.W, 
                              options.eta, options.alpha, options.gamma,
                              options.kappa, options.tau, options.scale,
                              options.adding_noise)
  if options.new_init:
    ohdp.new_init(c_train)

  print("setting up counters and log files.")

  iter = 0
  save_lag_counter = 0
  total_time = 0.0
  total_doc_count = 0
  split_doc_count = 0
  log_file = open("%s/log.dat" % result_directory, "w") 
  log_file.write("iteration time doc.count score word.count\n")

  if options.test_data_path is not None:
    test_log_file = open("%s/test-log.dat" % result_directory, "w") 
    test_log_file.write("iteration time doc.count score word.count score.split word.count.split\n")

  print("starting online variational inference.")
  while True:
    iter += 1
    if iter % 10 == 1:
      print("iteration: %09d" % iter)
    t0 = time.clock()

    # Sample the documents.
    batchsize = options.batchsize
    if options.seq_mode:
      c = read_stream_data(train_file, batchsize) 
      batchsize = c.num_docs
      if batchsize == 0:
        break
      docs = c.docs
    else:
      ids = random.sample(range(c_train.num_docs), batchsize)
      docs = [c_train.docs[id] for id in ids]

    total_doc_count += batchsize
    split_doc_count += batchsize

    # Do online inference and evaluate on the fly dataset
    score = ohdp.process_documents(docs, options.var_converge)
    total_time += time.clock() - t0
    log_file.write("%d %d %d %.5f\n" % (iter, total_time,
                    total_doc_count, score))
    log_file.flush()

    # Evaluate on the test data: fixed and folds
    if total_doc_count % options.save_lag == 0:
      if not options.fixed_lag and save_lag_counter < 10:
        save_lag_counter += 1
        options.save_lag = options.save_lag * 2

      # Save the model.
      ohdp.save_topics('%s/doc_count-%d.topics' %  (result_directory, total_doc_count))
      pickle.dump(ohdp, open('%s/doc_count-%d.model' % (result_directory, total_doc_count), 'wb'))

    # read another split.
    if not options.seq_mode:
      if split_doc_count > num_of_doc_each_split * options.pass_ratio and num_train_splits > 1:
        print("Loading a new split from the training data")
        split_doc_count = 0
        # cur_chosen_split = int(random.random() * num_train_splits)
        cur_chosen_split = (cur_chosen_split + 1) % num_train_splits
        cur_train_filename = train_filenames[cur_chosen_split]
        c_train = read_data(cur_train_filename)

    if (options.max_iter != -1 and iter >= options.max_iter) or (options.max_time !=-1 and total_time > options.max_time):
      break
  log_file.close()

  print("Saving the final model and topics.")
  ohdp.save_topics('%s/final.topics' %  result_directory)
  pickle.dump(ohdp, open('%s/final.model' % result_directory, 'wb'))

  if options.seq_mode:
    train_file.close()

  # Makeing final predictions.
  if options.test_data_path is not None:
    print("Making predictions.")
    labels_test = np.array([doc.ys for doc in c_test.docs])
    (_, preds, gammas_test) = ohdp.infer_only(c_test.docs, options.var_converge)
    print("HDP")
    for i in range(ohdp.num_responses()):
      report = classification_report(labels_test[:,i], preds[:,i])
      confusion = confusion_matrix(labels_test[:,i], preds[:,i])
      accuracy = accuracy_score(labels_test[:,i], preds[:,i])
      print("Accuracy rate : %f" % accuracy)
      print(report)    
      print(confusion)

    labels_train = np.array([doc.ys for doc in c_train.docs])
    (_, _, gammas_train) = ohdp.infer_only(c_train.docs, options.var_converge)
    print("Logistic Regression")
    for i in range(ohdp.num_responses()):
      clf = LogisticRegression()
      clf.fit(gammas_train, labels_train[:,i])
      preds = clf.predict(gammas_test)
      report = classification_report(labels_test[:,i], preds)
      confusion = confusion_matrix(labels_test[:,i], preds)
      accuracy = accuracy_score(labels_test[:,i], preds)
      print("Accuracy rate : %f" % accuracy)
      print(report)
      print(confusion)

if __name__ == '__main__':
  run_online_hdp()
