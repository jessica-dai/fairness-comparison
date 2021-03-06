import numpy as np
import csv

# preprocess
def preprocess_averages(file):
    read_data = read_csv(file)
    sep_data = alg_datatype_sep(read_data)
    for elt in sep_data:
        if len(sep_data[elt]) > 10:
            print("fixing " + elt)
            sep_data[elt] = fix_pgs(sep_data[elt])
            print(type(sep_data[elt]))
    rk = re_key(sep_data)
    
    return re_key_rows(analyze(rk)) # each row's last elements are mean, std

def preprocess_trials(file):
    read_data = read_csv(file)
    # print(len(read_data))
    sep_data = alg_datatype_sep(read_data)
    for elt in sep_data:
        if len(sep_data[elt]) > 10:
            # print("fixing " + elt)
            sep_data[elt] = fix_pgs(sep_data[elt])
    rk = re_key(sep_data)
    return re_key_trials(rk)

# read csv lines
def read_csv(file):
    # saves csv into indexable array & returns (single array)
    # Algorithm name,trial,data type,baseline DI Binary,post-alg DI Binary,baseline DI avg all,post-alg DI avg all,baseline CV,post-alg CV
    to_return = []
    with open(file, newline='') as curr_file:
        c_reader = csv.reader(curr_file)
        for row in c_reader:
            to_return.append(row)
    return to_return[1:]

# separate into different buckets for alg + datatype
def alg_datatype_sep(read_data):
    # returns dict of arrays where each elt in the list has the same alg name & data type
    buckets = {}
    for row in read_data:
        # print(row)
        key = str(row[0]) + '_' + str(row[2])
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(row)

    # print(len(buckets[list(buckets.keys())[0]]))
    # print(len(buckets))
    # print(type(buckets))
    return buckets

# fixes bucket for algs w param grid search
def fix_pgs(to_fix):
    # we want to separate this into *more!* buckets, one for each param
    param_bkts = {}

    # keeps track of baseline calculations/ makes sure they go to the right bucket 
    # we can do this because of the way the csv is produced -- not great style but w/e
    curr_trial = -1
    curr_di_b = 0
    curr_di_a = 0
    curr_cv = 0

    # for row in to_fix:
    for i in range(len(to_fix)):
        row = to_fix[i]
        # if row[-1] != " ": # this has the right baseline calculations for current trial
        # if i % 10 == 0:
        if int(row[1]) != curr_trial:
            curr_trial += 1
            curr_di_b = row[3]
            curr_di_a = row[5]
            curr_cv = row[7]
        elif int(row[1]) == curr_trial:
            reconstructed = [row[0], row[1], row[2], curr_di_b, row[3], curr_di_a, row[4], curr_cv, row[5]]
            # key = row[-2] # for non balanced 
            key = row[-1] # for balanced
            # print(row)
            if key not in param_bkts:
                param_bkts[key] = []
            param_bkts[key].append(reconstructed)

    print(len(param_bkts))
    return param_bkts

# for paramgridsearch buckets, changes them so the alg x param is an algorithm name
# returns dict where key is alg name, value is a 10 x 6 np array
def re_key(old_bkts):
    print("alsdjfk")
    bkts = {}
    print(len(old_bkts))
    ctr = 0
    for alg_name in old_bkts:
        bkt = old_bkts[alg_name]
        if type(bkt) == list: # i.e. non paramgridsearch ones
            bkts[alg_name] = np.array(bkt)[:,3:].astype(np.float)
            ctr+= 1
        else: # if the bucket is a dictionary
            print("s;djfalskdjfa;lskdjfsd")
            for param in bkt:
                print(param)
                lst = bkt[param]
                new_key = alg_name + '_' + str(param)
                bkts[new_key] = np.array(lst)[:,3:].astype(np.float)
                print(len(bkts))
    print(len(bkts))
    print("????")
    print(ctr)
    return bkts

# find average & diffs across trials..
def analyze(dict_of_results):
    analysis = {}
    for alg in dict_of_results:
        trials = dict_of_results[alg]
        # print("---------" + alg + " average diffs:---")
        diffs = trials[:,1] - trials[:,0]
        # means = np.mean(trials, axis=0)
        std = np.std(diffs)
        mean = np.mean(diffs)
        # print(means)
        # di_b_diff = means[1] - means[0]
        # di_a_diff = means[3] - means[2]
        # cv_diff = means[5] - means[4]
        # print("di binary: " + str(di_b_diff))
        # print("di avg: " + str(di_a_diff))
        # print("cv:" + str(cv_diff))
        # print("")
        # analysis[alg] = np.array([di_b_diff, di_a_diff, cv_diff])
        analysis[alg] = [mean, std]
    print(len(analysis))
    return analysis

def re_key_rows(old_bkts):
    readable_rows = []

    for alg in old_bkts:
        alg_details = str.split(alg, '_') # [alg name, data type, param if exists]
        if (len(alg_details) < 3):
            alg_details.append(' ')
        info = old_bkts[alg] # we only really care about 0 (baseline), 1 (DI binary), and diff
        alg_details = alg_details + info

        readable_rows.append(alg_details)

    print(len(readable_rows))
    return readable_rows

def re_key_trials(old_bkts):
    readable_rows = []

    for alg in old_bkts:
        alg_details = str.split(alg, '_') # [alg name, data type, param if exists]
        if (len(alg_details) < 3):
            alg_details.append(' ')
        trials = old_bkts[alg] # we only really care about 0 (baseline), 1 (DI binary), and diff
        
        if alg_details[1] != "numerical-binsensitive":
            continue
        for trial in trials:
            trial_data = alg_details + [trial[0], trial[1], trial[1] - trial[0]]
            readable_rows.append(trial_data)

    return readable_rows

def write_to_csv(rows, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main():

    # file = 'results/raw/ricci_Race_balanced.csv'
    file = 'results/raw/ricci_Race.csv'
    write_to_csv(preprocess_averages(file),'results/ricci_race_processed.csv')
    # write_to_csv(preprocess_trials(file), 'results/ricci_race_balanced_trials_processed.csv')
    # write_to_csv(preprocess_averages(file),'results/ricci_race_balanced_processed.csv')
    # files = ['results/ricci_Race.csv'] #, 'results/ricci_Race_balanced.csv']

    # analyzed = []
    # for file in files:
    #     print("analyzing " + file)
    #     pre = preprocess(file)
    #     analyzed.append(analyze(pre))

    # keys = analyzed[0].keys()

    # balanced_diffs = {}

    # for key in keys:
    #     # if key not in analyzed[1] or key not in analyzed[0]:
    #     #     continue
    #     # balanced - unbalanced
    #     # balanced_diffs[key] = np.subtract(analyzed[1][key], analyzed[0][key])
    #     print(key)
    #     print(analyzed[0][key])
    #     # print(balanced_diffs[key])
    #     print("")

    # print(len(analyzed[0]))

if __name__ == '__main__':
    main()
    