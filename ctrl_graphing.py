import numpy as np
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import argparse

"""
Some hard-coded things to loop over, to avoid unnecessary imports
"""
k_Ratios = [0.5, 0.57142857142857, 0.66666666666667, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0]
datasets = {
    'ricci': ['Race'],
    'adult': ['race', 'sex', 'race-sex'],
    'german': ['sex', 'age', 'sex-age'],
    'propublica-recidivism': ['sex', 'race', 'sex-race'],
    'propublica-violent-recidivism': ['sex', 'race', 'sex-race']
}

"""
TODO (intial from paper)
    1) different measures of fairness for a single alg/dataset pair (using stahel-donoho estimator)
        
        just make one graph for each k value and compare side-by-sides

        potential total # of graphs: algs*datasets*10

    4) stability (boxes of mean/stdev) for all algorithms on a single dataset
        a) one graph for each k value and compare side by sides
            datasets*10
        b) single dataset single algorithm all k values (color-code k values)
            algs*datasets
    5) algorithms with tunable parameters fairness x accuracy

        one graph for each k value and compare

"""

def load_dataset(dataset, sens):
    # access relevant data - we want to combine all of the files for the given dataset into a single df
    loaded_dataset = pd.DataFrame()
    for k in k_Ratios:
        filename = "results/" + str(dataset) + "_" + str(sens) + "_numerical-binsensitive_" + str(k) + ".csv"
        curr_dataset = pd.read_csv(filename)
        curr_dataset['k'] = k
        loaded_dataset = pd.concat([loaded_dataset, curr_dataset])
    return loaded_dataset

def scatter(singleK, dataset, sens, ax_x, ax_y):
    """
    scatter() is a general-purpose quick graph for a single dataset, given a single sensitive
    attribute; plots two values on the x- and y- axis, and can either do the same k over multiple
    algorithms, or the same algorithm over multiple ks. In general the latter gives better-looking 
    results. Used to generate the following:

        2) [sens]-calibration and [sens] TPR for all algorithms on a single dataset

        3) fairness (DIBinary) vs accuracy for all algorithms on a single dataset
            (for these graphs, also mark line)
            
    """

    loaded_dataset = load_dataset(dataset, sens)

    # params for graphing
    if (singleK):
        s_hue="algorithm"
        s_col="k"
        imgname = str(dataset) + "_" + str(sens) + "_" + str(ax_x) + "_" + str(ax_y)+ "_singleK" + ".png"
    else:
        s_hue = "k"
        s_col = "algorithm"
        imgname = str(dataset) + "_" + str(sens) + "_" + str(ax_x) + "_" + str(ax_y)+ "_singleAlg" + ".png"


    sns.relplot(x = ax_x, y = ax_y, hue=s_hue, col=s_col, col_wrap = 4, data=loaded_dataset).savefig(imgname)
    # plt.show()

def metric_over_k(dataset, sens, metric):
    """
    a single plot is for a single metric and single algorithm; 
    full img file has all algs for a single metric

    no real good results with this tbh

    x-axis: values of k (increasing) 
    y-axis: values of the metric (at every trial)
    """

    loaded_dataset = load_dataset(dataset, sens)

    imgname = str(dataset) + "_" + str(sens) + "_" + str(metric)

    sns.relplot(x = "k", y = metric, col="algorithm", col_wrap=4, data=loaded_dataset).savefig(imgname)

def avg_metric_over_k_line(dataset, sens):
    """
    plots averages of a select number of metrics over k
    """

    loaded_dataset = load_dataset(dataset, sens)
    # metrics
    metrics = ['accuracy', 'DIbinary', sens + '-calibration-', sens + '-accuracy', sens + '-TPR']
    aggregation_functions = {}
    for metric in metrics:
        aggregation_functions[metric] = 'mean'

    df_new = loaded_dataset.groupby(['algorithm', 'k']).aggregate(aggregation_functions).reset_index()
    df_new = df_new.melt(id_vars = ['algorithm', 'k'], value_vars = metrics)

    imgname = str(dataset) + "_" + str(sens) + "_" + "avgmetrics.png"

    sns.relplot(x="k", y ='value', hue='variable', kind='line', data=df_new).savefig(imgname)

def pretty_graph(dataset, sens, graphtype, ax_x, ax_y, xlim, ylim, alglist):
    """
    produces a SINGLE pretty graph (x/y labels, title, correct scaling)
    """

    loaded_dataset = load_dataset(dataset, sens)
    new_dataset = loaded_dataset[loaded_dataset['algorithm'].isin(alglist)]

    imgname = "pretty" + "_" + dataset + "_" + sens + "_" + graphtype
    
    sns.relplot(x = ax_x, y = ax_y, hue="k", col="algorithm", col_wrap = 3, data=new_dataset).set(xlim=xlim, ylim=ylim).savefig(imgname)
    # sns.scatterplot(x=ax_x, y=ax_y, hue="k", data=new_dataset).figure.savefig(imgname)
    # plot.set(xlim=xlim, ylim=ylim) # needs to be in the form (start, end)
    # plot.figure.savefig(imgname)

if __name__ == '__main__': 

    # dataset_algs = {
    #     'adult': ['LR', 'ZafarFairness', 'Kamishima', 'Feldman-SVM', 'Feldman-LR', 'Feldman-DecisionTree'],
    #     'german': ['LR', 'DecisionTree', 'GaussianNB', 'Feldman-GaussianNB', 'Feldman-LR', 'Feldman-DecisionTree'],
    #     'propublica-recidivism': ['SVM', 'DecisionTree', 'Calders', 'Kamishima', 'Feldman-LR', 'Feldman-DecisionTree'],
    #     'ricci': ['SVM', 'Calders', 'ZafarFairness', 'Feldman-SVM', 'Feldman-LR', 'Feldman-DecisionTree']
    # }

    # dataset_sens = {
    #     'adult': 'race', 'german': 'sex-age', 'propublica-recidivism': 'race', 'ricci': 'Race'
    # }

    # for dataset in dataset_algs:
    #     sens = dataset_sens[dataset]
    #     pretty_graph(dataset, sens, 'fairness', sens+'-TPR', sens+'-calibration-', None, None, dataset_algs[dataset])
    
    algs = ['SVM', 'GaussianNB', 'Feldman-SVM-DIavgall', 'Feldman-GaussianNB-DIavgall']
    
    for alg in algs:
        pretty_graph('adult', 'race', 'accuracy', 'DIbinary', 'accuracy', (0.5,0.9), (0.75, 0.88), [alg])
    
    print("done")