import fire
import os
import statistics
import sys

import pandas as pd
import numpy as np
from fairness import results
from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData, TRAINING_PERCENT
from fairness.algorithms.list import ALGORITHMS
from fairness.metrics.list import get_metrics
from fairness.benchmark import run_eval_alg, write_alg_results # rewritten: create_detailed_file

import results_writing_new # newly added
import warnings

from fairness.algorithms.ParamGridSearch import ParamGridSearch

TAGS = ['numerical-binsensitive']
NUM_TRIALS_DEFAULT = 10
SPLIT_PCT = 0.8 # this is not the proportion we're controlling for, just indicates on aggregate 80% of all available data is training etc

class ControlledProcessedData(ProcessedData):
    """
    This class allows for controlling the proportionality of training and testing splits. 

    The key, tunable constant is self.split_proportion ("k"), which represents "q/r"
        q: the proportion of the protected class in the training set ("available data")
        r: the proportion of the protected class in the testing set ("real world")
    
    This means that:
        k = 1 is perfect balance (q = r = prior)
        k > 1 is overrepresentation in the training set
        k < 1 is underrepresentation in the training set

    """

    def __init__(self, data_obj):
        self.data = data_obj
        self.dfs = dict((k, pd.read_csv(self.data.get_filename(k)))
                        for k in TAGS)
        self.data_size = len(self.dfs['numerical-binsensitive'])
        self.splits = dict((k, []) for k in TAGS)
        self.has_splits = False
        self.controlled_splits = self.splits
        self.has_controlled_splits = False
        self.split_proportion = 1 # default is perfect balance 
            # keeps track of the proportionality of the split.
            # set by set_proportion
       
    def set_proportion(self, prop):
        self.split_proportion = prop
        if (prop != 0):
            self.has_controlled_splits = True
    
    def get_protected_idx(self):
        pd.read_csv(self.data.get_filename(k))
    
    def create_controlled_train_test_splits(self, num, ctrl_attr, balance_outcomes = True):
        """
        num: number of trials (number of batches)
        ctrl_attr: the attribute to be controlled 
        # TODO balance_outcomes: flag for whether to ensure perfect proportionality of outcomes for each subset
        
        recall that this relies on self.split_proportion = q/r
            q = % protected training -> q/(q+r) percent of protected attribute in training
                -> q/(q+r) = k/(k+1)
            r = % protected testing -> rest in training

        """
        # if self.has_controlled_splits:
        #     print("hsdlkfjsd")
        #     return self.controlled_splits
        # print("hello")

        # numbers used in proportion calculations
        n_priv = len(np.where(self.dfs['numerical-binsensitive'][ctrl_attr] == 1)[0])  
            # count # of privileged examples of the dataset w/r/t current attribute
        prop = 4*self.split_proportion/(4*self.split_proportion + 1) # proportion of sens data to be in training

        if balance_outcomes: # adds class attribute to be split over
            ctrls = [ctrl_attr]
            ctrls.append(self.data.class_attr) 
            sensitive_vals = self.dfs['numerical-binsensitive'][ctrls].astype(str).apply(lambda x: '_'.join(x), axis=1).values
        else:
            sensitive_vals = self.dfs['numerical-binsensitive'][ctrl_attr].astype(str).apply(lambda x: '_'.join(x)).values
            # print(sensitive_vals) # this is representation of JUST the attributes to be split over for every data point
        sensitive_levels = pd.unique(sensitive_vals) # this is every possible combination of attributes to be split over
        
        non_priv = []
        priv = []
        for cur_sensitive in sensitive_levels:
            if cur_sensitive[0] == '0':
                non_priv.append(cur_sensitive)
            else:
                priv.append(cur_sensitive)

        for i in range(0, num): # for every trial
            # create empty lists for each  portion
            train_fraction = np.asarray([])
            test_fraction = np.asarray([])

            for nonp in non_priv: # deal w the non-priv class first
                print("nonp: " + nonp)
                c_attr_idx = np.where(sensitive_vals == nonp)[0] # find relevant indices
                np.random.shuffle(c_attr_idx) # and shuffle

                print("hi")
                print(len(c_attr_idx))

                # total_size = min(len(c_attr_idx), 0.8*self.data_size) # total number of sens examples to be using
                # split_ix = int(prop*total_size) 
                split_ix = int(prop*len(c_attr_idx))

                train_fraction = list(np.concatenate([train_fraction,c_attr_idx[:split_ix]]))
                print(len(train_fraction))
                test_fraction = list( np.concatenate([test_fraction,c_attr_idx[split_ix:]]))
                print(len(test_fraction))

            for p in priv: # fill in whatever's remaining
                print("p: " + p)

                c_attr_idx = np.where(sensitive_vals == p)[0]
                np.random.shuffle(c_attr_idx)

                print(len(c_attr_idx))

                # some algebra
                train_size = int((4*(len(test_fraction) + len(c_attr_idx)) - len(train_fraction))/5)
                test_size = len(c_attr_idx) - train_size
                # train_size = int(0.8*self.data_size) - len(train_fraction) # number to go in train
                # test_size = int(0.2*self.data_size) - len(test_fraction) #number to go in test

                if (train_size + test_size) > len(c_attr_idx):
                    return False # we could reduce OR just throw false here
                
                train_fraction = list(np.concatenate([train_fraction,c_attr_idx[:train_size]]))
                test_fraction = list( np.concatenate([test_fraction,c_attr_idx[train_size:]]))

            print("split finished")
            print(len(train_fraction))
            print(len(test_fraction))
            for (k, v) in self.dfs.items():
                train = self.dfs[k].iloc[train_fraction]
                test = self.dfs[k].iloc[test_fraction]
                self.controlled_splits[k].append((train, test))

        # print(self.balanced_splits.keys())
            
        self.has_controlled_splits = True
        return self.controlled_splits

def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    # print("Available algorithms:")
    # for a in result:
    #     print("  %s" % a)
    return result

def create_detailed_file(filename, dataset, sensitive_dict, tag):
    return results_writing_new.NewResultsFile(filename, dataset, sensitive_dict, tag)

def run(num_trials = NUM_TRIALS_DEFAULT, dataset = get_dataset_names(),
        algorithm = get_algorithm_names(), ctrl = 1):
    algorithms_to_run = algorithm

    print("Datasets: '%s'" % dataset)
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        processed_dataset = ControlledProcessedData(dataset_obj) 
        processed_dataset.set_proportion(ctrl) # do this after instantiating
        # moved train test split creation inside the attribute

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        for sensitive in all_sensitive_attributes: # sensitive is a string

            print("Sensitive attribute:" + sensitive)

            train_test_splits = processed_dataset.create_controlled_train_test_splits(num_trials, sensitive)
            if train_test_splits == False:
                continue # then this means K was wrong

            detailed_files = dict((k, create_detailed_file(
                                          dataset_obj.get_results_filename_ctrl(sensitive, k, ctrl),
                                          dataset_obj,
                                          processed_dataset.get_sensitive_values(k), k))
                                  for k in train_test_splits.keys())

            for algorithm in ALGORITHMS:
                if not algorithm.get_name() in algorithms_to_run:
                    continue

                print("    Algorithm: %s" % algorithm.get_name())
                print("       supported types: %s" % algorithm.get_supported_data_types())
                if algorithm.__class__ is ParamGridSearch:
                    param_files =  \
                        dict((k, create_detailed_file(
                                     dataset_obj.get_param_results_filename_ctrl(sensitive, k,
                                                                            algorithm.get_name(), ctrl),
                                     dataset_obj, processed_dataset.get_sensitive_values(k), k))
                          for k in train_test_splits.keys())
                for i in range(0, num_trials):
                    for supported_tag in ['numerical-binsensitive']: # algorithm.get_supported_data_types():
                        train = train_test_splits[supported_tag][i][0]
                        test = train_test_splits[supported_tag][i][1]
                        try:
                            params, results, param_results =  \
                                run_eval_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                             all_sensitive_attributes, sensitive, supported_tag)
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("Failed: %s" % e, file=sys.stderr)
                        else:
                            write_alg_results(detailed_files[supported_tag],
                                              algorithm.get_name(), params, i, results)
                            if algorithm.__class__ is ParamGridSearch:
                                for params, results in param_results:
                                    write_alg_results(param_files[supported_tag],
                                                      algorithm.get_name(), params, i, results)

            print("Results written to:")
            for supported_tag in algorithm.get_supported_data_types():
                print("    %s" % dataset_obj.get_results_filename(sensitive, supported_tag))

            for detailed_file in detailed_files.values():
                detailed_file.close()


if __name__ == '__main__': 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        run(dataset = ['ricci'], ctrl = 1)

    # p_data = ControlledProcessedData(DATASETS[0])
    # p_data.set_proportion(1.3)
    # train_test_splits = p_data.create_controlled_train_test_splits(10, 'Race')
    # if train_test_splits == False:
    #     print("sigh")
    # print("hi")
    # print(len(train_test_splits['numerical-binsensitive'][9][1]))
    # # train, test = train_test_splits['numerical-binsensitive']
    # print(len(train))
    # print(len(test))

    # print(train[0])