import os
from corpus import *
import onlinehdp
import pickle
from optparse import OptionParser
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
np = onlinehdp.np

def parse_args():
  parser = OptionParser()
  parser.set_defaults(kappa=0.9, tau=1., batchsize=500, var_converge=0.0001,
                      corpus_name=None, data_path=None, test_data_path=None, 
                      test_data_path_in_folds=None, directory=None,
                      seq_mode=False)

  parser.add_option("--kappa", type="float", dest="kappa",
                    help="learning rate [0.9]")
  parser.add_option("--tau", type="float", dest="tau",
                    help="slow down [1.0]")
  parser.add_option("--batchsize", type="int", dest="batchsize",
                    help="batch size [500]")
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
  parser.add_option("--seq_mode", action="store_true", dest="seq_mode",
                    help="processing the data in the sequential mode")
  
  (options, args) = parser.parse_args()
  return options 

def run_online_hdp():
  # Command line options.
  options = parse_args()

  # Set the random seed.
  if not options.seq_mode:
    train_filenames = glob(options.data_path)
    train_filenames.sort()
    # This is used to determine when we reload some another split.
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
  ohdp = pickle.load(open('%s/final.model' % result_directory, 'rb'), encoding='latin1')
  ohdp.print_model()

  alpha, _ = ohdp.hdp_to_lda()
  f = open('%s/final.alpha' % result_directory, 'w')
  line = ','.join([str(x) for x in alpha])
  f.write(line + '\n')

  (_, _, gammas_train) = ohdp.infer_only(c_train.docs, options.var_converge)
  labels_train = np.array([doc.ys for doc in c_train.docs])
  f = open('%s/final.gamma' % result_directory, 'w')  
  for gamma in gammas_train:
    line = ','.join([str(x) for x in gamma])  
    f.write(line + '\n')

  f = open('%s/final.eta' % result_directory, 'w')
  for response in ohdp.m_responses:
    line = ','.join([str(x) for x in response.mu])
    f.write(line + '\n')  
    

  # Making final predictions.
  if options.test_data_path is not None:
    print("Making predictions.")
    labels_test = np.array([doc.ys for doc in c_test.docs])
    (_, _, gammas_test) = ohdp.infer_only(c_test.docs, options.var_converge)
    # print("HDP")
    # for i in range(ohdp.num_responses()):
    #   report = classification_report(labels_test[:,i], preds[:,i])
    #   confusion = confusion_matrix(labels_test[:,i], preds[:,i])
    #   accuracy = accuracy_score(labels_test[:,i], preds[:,i])
    #   print("Accuracy rate : %f" % accuracy)
    #   print(report)    
    #   print(confusion)
#    print(hamming_loss(labels_test[:,:-1], preds[:,:-1]))

    print("Logistic Regression")
    preds = np.zeros((len(c_test.docs), 256))
    for i in range(256):
      clf = LogisticRegression()
      clf.fit(gammas_train, labels_train[:,i])
      preds[:,i] = clf.predict(gammas_test)
      report = classification_report(labels_test[:,i], preds[:,i])
      confusion = confusion_matrix(labels_test[:,i], preds[:,i])
      accuracy = accuracy_score(labels_test[:,i], preds[:,i])
      print("Accuracy rate : %f" % accuracy)
      print(report)
      print(confusion)
#    print(hamming_loss(labels_test[:,:-1], preds[:,:-1]))

if __name__ == '__main__':
  run_online_hdp()
