import fire
import os
import statistics
import sys
import csv

import pandas as pd
import numpy as np
from fairness import results
from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData, TAGS, TRAINING_PERCENT
from fairness.algorithms.list import ALGORITHMS
from fairness.metrics.list import get_metrics
from fairness.benchmark import create_detailed_file, run_eval_alg, write_alg_results
# from fairness.

from fairness.algorithms.ParamGridSearch import ParamGridSearch

NUM_TRIALS_DEFAULT = 10

# TAGS = ["original", "numerical", "numerical-binsensitive", "categorical-binsensitive"]
# TRAINING_PERCENT = 2.0 / 3.0

class BalancedProcessedData(ProcessedData):
    def __init__(self, data_obj):
        self.data = data_obj
        self.dfs = dict((k, pd.read_csv(self.data.get_filename(k)))
                        for k in TAGS)
        self.splits = dict((k, []) for k in TAGS)
        self.has_splits = False
        self.balanced_splits = self.splits
        self.has_balanced_splits = False
       
    
    def get_protected_idx(self):
        pd.read_csv(self.data.get_filename(k))
    
    
    def create_balanced_train_test_splits(self, num):
        if self.has_balanced_splits:
            return self.balanced_splits
        
                # get all the variables to balanced over
        balance_attr = self.data.sensitive_attrs.copy()
        balance_attr.append(self.data.class_attr)

        # merge them all into a single variable
        sensitive_vals = self.dfs['original'][balance_attr].astype(str).apply(lambda x: '_'.join(x), axis=1).values

        sensitive_levels = pd.unique(sensitive_vals)
        
        
        for i in range(0, num):
            # we first shuffle a list of indices so that each subprocessed data
            # is split consistently
            

            # create empty lists for each  portion
            train_fraction = np.asarray([])
            test_fraction = np.asarray([])

            for cur_sensitive in sensitive_levels:
                # randomly split each value of the protected class 
                c_attr_idx = np.where(sensitive_vals == cur_sensitive)[0]
                np.random.shuffle(c_attr_idx)

                split_ix = int(len(c_attr_idx) * .8)
                train_fraction = list(np.concatenate([train_fraction,c_attr_idx[:split_ix]]))
                test_fraction = list( np.concatenate([test_fraction,c_attr_idx[split_ix:]]))

#                 for cur_sensitive in sensitive_levels:
#                     # randomly split each value of the protected class 
#                     c_attr_idx = np.where(sensitive_vals.values == cur_sensitive)[0]
#                     np.random.shuffle(c_attr_idx)

#                     split_ix = int(len(c_attr_idx) * TRAINING_PERCENT)
#                     train_fraction = np.concatenate([train_fraction,c_attr_idx[:split_ix]])
#                     test_fraction = np.concatenate([test_fraction,c_attr_idx[split_ix:]])
            # TODO if multiple balances, get list for each interesection and randomly split each of those
            # appned all trains together and all tests together
            
                       
            for (k, v) in self.dfs.items():
                train = self.dfs[k].iloc[train_fraction]
                test = self.dfs[k].iloc[test_fraction]
                self.balanced_splits[k].append((train, test))

        print(self.balanced_splits.keys())
            
        self.has_balanced_splits = True
        return self.balanced_splits

def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

def run(num_trials = NUM_TRIALS_DEFAULT, dataset = get_dataset_names(),
        algorithm = get_algorithm_names()):
    algorithms_to_run = algorithm
    print("hello")
    print("Datasets: '%s'" % dataset)
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        processed_dataset = BalancedProcessedData(dataset_obj)
        train_test_splits = processed_dataset.create_balanced_train_test_splits(num_trials)

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        for sensitive in all_sensitive_attributes:

            print("Sensitive attribute:" + sensitive)

            # detailed_files = dict((k, create_detailed_file(
            #                               dataset_obj.get_results_filename(sensitive, k),
            #                               dataset_obj,
            #                               processed_dataset.get_sensitive_values(k), k))
            #                       for k in train_test_splits.keys())

            data_to_write = [] # ADDED BY JESS
            # add heading row
            data_to_write.append(["Algorithm name", 
                                    "trial", 
                                    "data type", 
                                    "baseline DI Binary", 
                                    "post-alg DI Binary",
                                    "baseline DI avg all",
                                    "post-alg DI avg all",
                                    "baseline CV",
                                    "post-alg CV"
                                    ])

            for algorithm in ALGORITHMS:
                if not algorithm.get_name() in algorithms_to_run:
                    continue

                print("    Algorithm: %s" % algorithm.get_name())
                print("       supported types: %s" % algorithm.get_supported_data_types())
                # if algorithm.__class__ is ParamGridSearch:
                #     param_files =  \
                #         dict((k, create_detailed_file(
                #                      dataset_obj.get_param_results_filename(sensitive, k,
                #                                                             algorithm.get_name()),
                #                      dataset_obj, processed_dataset.get_sensitive_values(k), k))
                #           for k in train_test_splits.keys())
                for i in range(0, num_trials):
                    for supported_tag in algorithm.get_supported_data_types():
                        train, test = train_test_splits[supported_tag][i]
                        try:
                            params, results, param_results =  \
                                run_eval_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                             all_sensitive_attributes, sensitive, supported_tag)
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("Failed: %s" % e, file=sys.stderr)
                        else:
                            row_starter = [algorithm.get_name(), i, str(supported_tag)]
                            data_to_write.append(row_starter + results)
                            # write_alg_results(detailed_files[supported_tag],
                            #                   algorithm.get_name(), params, i, results)
                            if algorithm.__class__ is ParamGridSearch:
                                for params, results in param_results:
                                    row_end = []
                                    for (k,v) in params.items():
                                        row_end.append(k)
                                        row_end.append(v)
                                    
                                    data_to_write.append(row_starter + results + row_end)

                                    # write_alg_results(param_files[supported_tag],
                                    #                   algorithm.get_name(), params, i, results)

            new_filename = dataset_obj.get_dataset_name() + "_" + sensitive + "_balanced_FULL.csv"

            with open("results/"+new_filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data_to_write)

            print("results written!")

            # print("Results written to:")
            # for supported_tag in algorithm.get_supported_data_types():
            #     print("    %s" % dataset_obj.get_results_filename(sensitive, supported_tag))

            # for detailed_file in detailed_files.values():
            #     detailed_file.close()

if __name__ == '__main__': 
    run(dataset=['ricci'])