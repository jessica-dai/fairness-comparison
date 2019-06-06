import pathlib
import os
import tempfile
import shutil

from fairness.metrics.list import get_metrics

##############################################################################

def get_metrics_list(dataset, sensitive_dict, tag):
    return [metric.get_name() for metric in get_metrics(dataset, sensitive_dict, tag)]

def get_detailed_metrics_header(dataset, sensitive_dict, tag):
    return ','.join(['algorithm', 'params', 'run-id'] + get_metrics_list(dataset, sensitive_dict, tag))
    
class NewResultsFile(object):

    def __init__(self, filename, dataset, sensitive_dict, tag):
        self.filename = filename
        self.dataset = dataset
        self.sensitive_dict = sensitive_dict
        self.tag = tag
        self.handle = self.create_new_file() # self.handle is the central file object

    def create_new_file(self):
        
        f = open(self.filename, 'w') # zz create a new file (not temporary)
        f.write(get_detailed_metrics_header(
            self.dataset, self.sensitive_dict, self.tag) + '\n')
        return f

    def write(self, *args):
        self.handle.write(*args)
        self.handle.flush()
        os.fsync(self.handle.fileno()) 
        # zz TODO figure out why/whether fsync is needed -- seems like right now is fine 
        # note that it won't work when run in onedrive

    def close(self):

        # i just dont really know why they needed to copy over to a new file??
        self.handle.close()

        # new_file = open(self.filename, "r")
        # new_columns = new_file.readline().strip().split(',')
        # new_rows = new_file.readlines()

        # try:
        #     old_file = open(self.filename, "r")
        #     old_columns = old_file.readline().strip().split(',')
        #     old_rows = old_file.readlines()
        # except FileNotFoundError:
        #     old_columns = new_columns[:3] # copy the key columns
        #     old_rows = []

        # final_columns = set(old_columns).union(set(new_columns))
        
        # # FIXME: here we cross our fingers that parameters don't have "," in them.
        # def indexed_rows(rows, column_names):
        #     result = {}
        #     for row in rows:
        #         entries = row.strip().split(',')
        #         result[tuple(entries[:3])] = dict(
        #             (entry_name, entry)
        #             for (entry_name, entry) in
        #             zip(column_names, entries))
        #     return result

        # old_indexed_rows = indexed_rows(old_rows, old_columns)
        # # now we merge the rows onto the old file.
        # for (key, value_dict) in indexed_rows(new_rows, new_columns).items():
        #     for (value_name, value) in value_dict.items():
        #         old_indexed_rows.setdefault(key, {})[value_name] = value

        # fd, final_tempname = tempfile.mkstemp()
        # os.close(fd)
        # final_file = open(final_tempname, "w")
        # final_columns_list = ["algorithm", "params", "run-id"] + \
        #     sorted(list(final_columns.difference(set(["algorithm", "params", "run-id"]))))
        # final_file.write(",".join(final_columns_list) + "\n")
        # for row_dict in old_indexed_rows.values():
        #     row = ",".join(list(row_dict.get(l, "") for l in final_columns_list))
        #     final_file.write(row + "\n")
        # final_file.close()
        # shutil.move(final_tempname, self.filename)
