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

def run():

    algorithms_to_run = get_algorithm_names()
    dataset = get_dataset_names()

    baseline_metrics = [DIBinary(), DIAvgAll(), CV()]

    rows_to_write = []

    print("Datasets: '%s'" % dataset)

    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())
        
        # n_trials = 1 

        processed_dataset = ProcessedData(dataset_obj)
        train_test_splits = processed_dataset.create_train_test_splits(1)

        train, test = train_test_splits["original"][0]
        # get the actual classifications and sensitive attributes
        actual = test[dataset_obj.get_class_attribute()].values.tolist()

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()

        dict_sensitive_lists = {}
        for sens in all_sensitive_attributes:
            dict_sensitive_lists[sens] = test[sens].values.tolist()

        for sensitive in all_sensitive_attributes:

            # this is a single row to write (3 possible baselines: DIBinary, DIAvgAll, CV)
            row = [dataset_obj.get_dataset_name(), str(sensitive)]

            # how to handle "tag"
            tag = None
            privileged_vals = dataset_obj.get_privileged_class_names_with_joint(tag)
            positive_val = dataset_obj.get_positive_class_val(tag)

            for metric in baseline_metrics:
                baseline = metric.calc(None, actual, dict_sensitive_lists, 
                    sensitive, privileged_vals, positive_val)
                row.append(baseline)

            # calculate priors
            df = processed_dataset.get_dataframe('numerical-binsensitive')
            total = len(df.index)

            print(df)

            total_priv = df.sum()[sens]

            prior = 1.0 - float(total_priv)/float(total)

            row.append(prior)

            rows_to_write.append(row)
    
    with open("baseline_list.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_write)


if __name__ == '__main__': 
    run()
