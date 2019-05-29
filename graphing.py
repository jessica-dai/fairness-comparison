import numpy as np 
import matplotlib.pyplot as plt 

from results_analysis import preprocess_averages, preprocess_trials

# data = np.array(preprocess_averages('results/raw/ricci_Race.csv'))
data = np.array(preprocess_averages('results/raw/ricci_Race_balanced.csv'))
# data_trials = np.array(preprocess_trials('results/raw/ricci_Race.csv'))
# data_trials = np.array(preprocess_trials('results/raw/ricci_Race_balanced.csv'))

def feldman_params():
    params = np.array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , \
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ]).astype(np.float)

    feld_svm = data[76:97,-2].astype(np.float) # best params 0.25 
    f_s_std = data[76:97, -1].astype(np.float)/3.162
    feld_nb = data[160:181,-2].astype(np.float)
    f_n_std = data[160:181,-1].astype(np.float)/3.162 # best params 0.8

    ind = np.arange(21)
    width = 0.4
    
    fig, ax = plt.subplots()
    svm = ax.bar(ind - width/2, feld_svm, width, color='SkyBlue', label='SVM', yerr=f_s_std, error_kw={'elinewidth':0.5})
    nb = ax.bar(ind + width/2, feld_nb, width, color='Violet', label='NB', yerr=f_n_std, error_kw={'elinewidth':0.5})

    ax.set_ylabel('DI difference from pre-alg priors')
    ax.set_xlabel('tradeoff parameter value')
    ax.set_title('feldman parameter tuning')
    ax.set_xticks(ind)
    ax.set_xticklabels(params, fontsize=8)
    ax.legend()
    ax.axhline(0, color='gray')
    ax.axhline(0.3, color='red', linewidth=0.5)
    ax.axhline(0.5, color='orange', linewidth=0.5)

    plt.show()
    # plt.savefig('feldman_params.png')

def kamishima_params():
    params = [0, 1, 2, 3, 4, 5, 10,15,20, 30, 40, 50, 100, 150,200, 250, 300]
    # best params 150, 200, 250
    
    kam_diffs = data[30:47,-2].astype(np.float)
    k_std = data[30:47, -1].astype(np.float)/3.162

    ind = np.arange(17)
    width = 0.3

    fig, ax = plt.subplots()
    diffs = ax.bar(ind, kam_diffs, width, color='SkyBlue', yerr=k_std, error_kw={'elinewidth':0.5})

    ax.set_ylabel('DI difference from pre-alg priors')
    ax.set_xlabel('tradeoff parameter value')
    ax.set_title('kamishima parameter tuning')
    ax.set_xticks(ind)
    ax.set_xticklabels(params, fontsize=8)
    # ax.legend()
    ax.axhline(0, color='gray')
    ax.axhline(0.3, color='red', linewidth=0.5)
    ax.axhline(0.5, color='orange', linewidth=0.5)

    plt.show()
    # plt.savefig('kamishima_params.png')

def all_diffs():

    # optimal params
    kam_diffs = data[43,-2].astype(np.float)
    k_err = data[43,-1].astype(np.float)
    feld_svm = data[81,-2].astype(np.float)
    fs_err = data[81, -1].astype(np.float)
    feld_nb = data[176,-2].astype(np.float)
    fn_err = data[176,-1].astype(np.float)

    generic_svm = 0.260150926
    g_s_err = 0.29830287
    generic_nb = -0.090131859
    g_n_err = 0.109851112
    generic_LR = -0.009286442
    g_lr_err = 0.023930817


    # calders = 2.226382824

    zafar_fair = 0.655102561
    z_err = 0.268855959

    # zafar_acc = 0.022525374

    # inds (cluster similar types of algorithms together)
    # svm, nb, LR // feldman svm, feldman nb //  kam zafar
    ind = np.array([0, 0.4, 0.8, 1.4, 1.8, 2.4, 2.8])
    to_graph = [generic_svm, generic_nb, generic_LR, feld_svm, feld_nb, kam_diffs, zafar_fair]
    params = ["baseline algorithms", "", "", "feldman (preprocessing)",  "", "", "algorithm constraints"]
    labels = ["SVM generic", "NB generic", "LR generic", "SVM feldman", "NB feldman", "kamishima", "zafar"]
    errors = [g_s_err, g_n_err, g_lr_err, fs_err, fn_err, k_err, z_err]

    width = 0.3

    fig, ax = plt.subplots()
    
    for i in range(len(ind)):
        ax.bar(ind[i], to_graph[i], width, label=labels[i], yerr = errors[i]/3.162, error_kw={'elinewidth':0.5})

    ax.set_ylabel('DI difference from pre-alg priors')
    # ax.set_title('comparison across strategies')
    ax.set_title('balanced comparison across strategies')
    ax.set_xticks(ind)
    ax.set_xticklabels(params)
    ax.legend()
    ax.axhline(0, color='gray')
    ax.axhline(0.3, color='red', linewidth=0.5)
    ax.axhline(0.5, color='orange', linewidth=0.5)

    plt.show()
    # plt.savefig('all_strategies.png')

def di_variance():
    # labels = ["SVM generic", "SVM feldman", "NB generic", "NB feldman", "calders", "LR generic", "kamishima", "zafar"]
    labels = ["SVM generic", "SVM feldman", "NB generic", "NB feldman", "LR generic", "kamishima", "zafar"]

    svm_generic = data_trials[:10,-2].astype(np.float)
    svm_feldman = data_trials[520:530, -2].astype(np.float) # param = 0.25

    nb_generic = data_trials[10:20, -2].astype(np.float)
    nb_feldman = data_trials[1260:1270, -2].astype(np.float) # param = 0.8
    # calders = data_trials[50:60, -2].astype(np.float)

    lr_generic = data_trials[20:30, -2].astype(np.float)
    kamishima = data_trials[220:230,-2].astype(np.float) # param = 150
    zafar = data_trials[70:80, -2].astype(np.float)


    # to_plot = [svm_generic, svm_feldman, nb_generic, nb_feldman, calders, lr_generic, kamishima, zafar]
    to_plot = [svm_generic, svm_feldman, nb_generic, nb_feldman, lr_generic, kamishima, zafar]

    fig, ax = plt.subplots()

    for i in range(len(labels)):
        y = np.full([10], 7-i)
        x = to_plot[i]
        ax.scatter(x, y, label=labels[i])

    ax.legend()
    ax.set_title('balanced DI variance across trials')
    ax.set_xlabel('post-algorithm disparate impact')
    ax.axvline(0.5, color='gray')
    ax.axvline(0.8, color='red', linewidth=0.5)
    ax.axvline(1.0, color='orange', linewidth=0.5)

    # plt.show()
    plt.savefig('graphs/balanced_di_variance.png')
    
def main():
    print("hello")
    # print(len(data))

    # feldman_params()
    # kamishima_params()
    all_diffs()
    # di_variance()

if __name__ == '__main__':
    main()