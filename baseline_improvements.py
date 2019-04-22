import fire
import os
import statistics
import sys
import csv

import pandas as pd 
import numpy as np

from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.algorithms.list import ALGORITHMS
from fairness.benchmark import create_detailed_file, get_algorithm_names, run_alg, write_alg_results # run_eval_alg, 
from balanced_splits import BalancedProcessedData
from fairness.data.objects.ProcessedData import ProcessedData
from fairness.algorithms.ParamGridSearch import ParamGridSearch
from fairness.metrics.CV import CV
from fairness.metrics.DIAvgAll import DIAvgAll
from fairness.metrics.DIBinary import DIBinary

NUM_TRIALS_DEFAULT = 10

def get_metrics(dataset, sensitive_dict, tag): 

    # METRICS = [ Accuracy(), TPR(), TNR(), BCR(), MCC(),        # accuracy metrics
    #         DIBinary(), DIAvgAll(), CV(),                  # fairness metrics
    #         SensitiveMetric(Accuracy), SensitiveMetric(TPR), SensitiveMetric(TNR),
    #         SensitiveMetric(FPR), SensitiveMetric(FNR),
    #         SensitiveMetric(CalibrationPos), SensitiveMetric(CalibrationNeg) ]

    baseline_metrics = [DIBinary(), DIAvgAll(), CV()]
    metrics = []

    for metric in baseline_metrics:
        metrics += metric.expand_per_dataset(dataset, sensitive_dict, tag)

    return metrics


def run_eval_alg(algorithm, train, test, dataset, processed_data, all_sensitive_attributes,
                 single_sensitive, tag):

    """
    changes made from original: added a run of the metric on the raw data
    """
    privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
    positive_val = dataset.get_positive_class_val(tag)

    # get the actual classifications and sensitive attributes
    actual = test[dataset.get_class_attribute()].values.tolist()
    sensitive = test[single_sensitive].values.tolist()

    # params is a dict mapping parameter names to default values
    predicted, params, predictions_list =  \
        run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
                privileged_vals, positive_val)

    # make dictionary mapping sensitive names to sensitive attr test data lists
    dict_sensitive_lists = {}
    for sens in all_sensitive_attributes:
        dict_sensitive_lists[sens] = test[sens].values.tolist()

    sensitive_dict = processed_data.get_sensitive_values(tag)
    one_run_results = []
    # baseline_results = []

    for metric in get_metrics(dataset, sensitive_dict, tag):
        baseline = metric.calc(None, actual, dict_sensitive_lists, 
                            single_sensitive, privileged_vals, positive_val)
        one_run_results.append(baseline)
        result = metric.calc(actual, predicted, dict_sensitive_lists, single_sensitive,
                             privileged_vals, positive_val)
        print(metric.get_name() + " baseline: " + str(baseline))
        print(metric.get_name() + " after algorithm: " + str(result))
        one_run_results.append(result)

    # handling the set of predictions returned by ParamGridSearch
    results_lol = []
    if len(predictions_list) > 0:
        for param_name, param_val, predictions in predictions_list:
            params_dict = { param_name : param_val }
            results = []
            for metric in get_metrics(dataset, sensitive_dict, tag):
                result = metric.calc(actual, predictions, dict_sensitive_lists, single_sensitive,
                                     privileged_vals, positive_val)
                results.append(result)
            results_lol.append( (params_dict, results) )

    return params, one_run_results, results_lol

def run(num_trials = NUM_TRIALS_DEFAULT, dataset = get_dataset_names(),
        algorithm = get_algorithm_names(), balanced = False):
    
    """
    either balanced or unbalanced based on flag.

    note: I originally was thinking of rewriting the for loop to be metric->algorithm instead of 
    algorithm->metric like it is now, but decided just to modify run_eval_alg  -- while this 
    means that eval_data is called once for each algorithm which is a little inefficient, I thought
    it would be better to preserve as much original code structure as possible. (obviously can change
    more if that would be better)
    """
    algorithms_to_run = algorithm

    print("Datasets: '%s'" % dataset)

    # iterate through datasets
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        if (balanced): 
            print("\n Processing with balanced splits: \n")
            processed_dataset = BalancedProcessedData(dataset_obj) 
            train_test_splits = processed_dataset.create_balanced_train_test_splits(num_trials)
        else:
            print("\n Processing with random splits: \n")
            processed_dataset = ProcessedData(dataset_obj)
            train_test_splits = processed_dataset.create_train_test_splits(num_trials)


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
                    print("trial " + str(i))
                    for supported_tag in algorithm.get_supported_data_types():
                        print("supported tag: " + str(supported_tag))
                        train, test = train_test_splits[supported_tag][i]
                        try:
                            params, results, param_results =  \
                                run_eval_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                             all_sensitive_attributes, sensitive, supported_tag)
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("something went wrong")
                            # print("Failed: %s" % e, file=sys.stderr)
                        else:
                            # write_alg_results(detailed_files[supported_tag],
                            #                   algorithm.get_name(), params, i, results)

                            row_starter = [algorithm.get_name(), i, str(supported_tag)]
                            data_to_write.append(row_starter + results)

                            if algorithm.__class__ is ParamGridSearch:
                                for params, results in param_results:
                                    # TODO fix, this is super super hacky
                                    row_end = []
                                    for (k,v) in params.items():
                                        row_end.append(k)
                                        row_end.append(v)
                                    
                                    data_to_write.append(row_starter + results + row_end)

                                    # write_alg_results(param_files[supported_tag],
                                    #                   algorithm.get_name(), params, i, results)
            
            if (balanced):
                new_filename = dataset_obj.get_dataset_name() + "_" + sensitive + "_balanced.csv"
            else:
                new_filename = dataset_obj.get_dataset_name() + "_" + sensitive + ".csv"

            # create file
            with open("results/"+new_filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data_to_write)


            # print("Results written to:")
            # for supported_tag in algorithm.get_supported_data_types():
            #     print("    %s" % dataset_obj.get_results_filename(sensitive, supported_tag))

            # for detailed_file in detailed_files.values():
            #     detailed_file.close()

if __name__ == '__main__': 
    run(balanced = True)
    run()