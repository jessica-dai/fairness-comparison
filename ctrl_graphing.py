import numpy as np
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import argparse

from fairness.data.objects.list import DATASETS, get_dataset_names

k_Ratios = [0.5, 0.57142857142857, 0.66666666666667, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0]

"""
    1) different measures of fairness for a single alg/dataset pair (using stahel-donoho estimator)
        
        just make one graph for each k value and compare side-by-sides

        potential total # of graphs: algs*datasets*10
"""
"""
    2) [sens]-calibration and [sens] TPR for all algorithms on a single dataset

        a) one graph for each k value and compare side by sides
            datasets*10
        b) single dataset single algorithm all k values (color-code k values)
            algs*datasets
"""
def cal_vs_TPR(singleK = True, dataset, sens, ax_x, ax_y):

    # access relevant data - we want to combine all of the files for the given dataset into a single df
    loaded_dataset = pd.DataFrame()
    for k in k_Ratios:
        filename = str(dataset) + "_" + str(sens) + "_numerical-binsensitive_" + str(k) + ".csv"
        curr_dataset = pd.read_csv(filename)
        curr_dataset['k'] = k
        loaded_dataset = pd.concat([loaded_dataset, curr_dataset])

    # params for graphing
    if (singleK):
        s_hue="algorithm"
        s_col="k"
        imgname = str(dataset) + "_" + "singleK" + ".png"
    else:
        s_hue = "k"
        s_col = "algorithm"
        imgname = str(dataset) + "_" + "singleAlg" + ".png"
    
    # x = "Race-TPR"
    # y = "Race-calibration-"

    sns.relplot(x = ax_x, y = ax_y, hue=s_hue, col=s_col, data=loaded_dataset).savefig(imgname)
    # plt.show()

def test(algname):
    print("sldkfjs;ldkfjsd" + str(algname))

"""
    3) fairness (DIBinary) vs accuracy for all algorithms on a single dataset

        a) one graph for each k value and compare side by sides
            datasets*10
        b) single dataset single algorithm all k values (color-code k values)
            algs*datasets

    4) stability (boxes of mean/stdev) for all algorithms on a single dataset
        a) one graph for each k value and compare side by sides
            datasets*10
        b) single dataset single algorithm all k values (color-code k values)
            algs*datasets
    
    5) algorithms with tunable parameters fairness x accuracy

        one graph for each k value and comp
"""

if __name__ == '__main__': 

    print("?")
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help = "name of graphing algorithm")
    parser.add_argument("--dataset", help = "dataset to graph")
    args = parser.parse_args()

    sns.set()

    print(args.graph)

    eval(args.graph)(args.dataset)

    print("hello")