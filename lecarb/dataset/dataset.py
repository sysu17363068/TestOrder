import os
import copy
import logging
import pickle
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Tuple
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"..")))
print(sys.path)

from lecarb.constants import DATA_ROOT, PKL_PROTO
from lecarb.dtypes import is_categorical
print(DATA_ROOT)
L = logging.getLogger(__name__)

from sklearn.cluster import KMeans

def kmeans_grouping(dictionary, num_clusters):
    # 获取所有的字典值并转换为列表
    values = list(dictionary.values())
    # 将列表转换为数组并进行形状调整
    data = np.array(values).reshape(-1, 1)

    # 初始化k-means模型并进行训练
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    # 获取聚类标签
    labels = kmeans.labels_

    # 将字典按照聚类标签分组
    groups = {}
    for i, label in enumerate(labels):
        if label not in groups:
            groups[label] = {}
        key = list(dictionary.keys())[i]
        value = dictionary[key]
        groups[label][key] = value

    return groups



class Column(object):
    def __init__(self, name, data):
        self.name = name
        self.dtype = data.dtype

        # parse vocabulary
        self.vocab, self.has_nan = self.__parse_vocab(data)
        self.vocab_size = len(self.vocab)
        self.minval = self.vocab[1] if self.has_nan else self.vocab[0]
        self.maxval = self.vocab[-1]

    def __repr__(self):
        return f'Column({self.name}, type={self.dtype}, vocab size={self.vocab_size}, min={self.minval}, max={self.maxval}, has NaN={self.has_nan})'

    def __parse_vocab(self, data):
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(data)
        contains_nan = np.any(is_nan)
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(data[~is_nan]))
        if contains_nan:
            vs = np.insert(vs, 0, np.nan)
        return vs, contains_nan

    def discretize(self, data):
        """Transforms data values into integers using a Column's vocabulary"""

        # pd.Categorical() does not allow categories be passed in an array
        # containing np.nan.  It makes it a special case to return code -1
        # for NaN values.
        if self.has_nan:
            bin_ids = pd.Categorical(data, categories=self.vocab[1:]).codes
            # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
            # add 1 to everybody
            bin_ids = bin_ids + 1
        else:
            # This column has no nan or nat values.
            bin_ids = pd.Categorical(data, categories=self.vocab).codes

        bin_ids = bin_ids.astype(np.int32, copy=False)
        assert (bin_ids >= 0).all(), (self, data, bin_ids)
        return bin_ids

    def normalize(self, data):
        """Normalize data to range [0, 1]"""
        minval = self.minval
        maxval = self.maxval
        # if column is not numerical, use descretized value
        if is_categorical(self.dtype):
            data = self.discretize(data)
            minval = 0
            maxval = self.vocab_size - 1
        data = np.array(data, dtype=np.float32)
        if minval >= maxval:
            L.warning(f"column {self.name} has min value {minval} >= max value{maxval}")
            return np.zeros(len(data)).astype(np.float32)
        val_norm = (data - minval) / (maxval - minval)
        return val_norm.astype(np.float32)

class Table(object):
    def __init__(self, dataset, version):
        self.dataset = dataset
        self.version = version
        self.name = f"{self.dataset}_{self.version}"
        L.info(f"start building data {self.name}...")

        # load data
        self.data = pd.read_pickle(DATA_ROOT / self.dataset / f"{self.version}.pkl")
        self.data_size_mb = self.data.values.nbytes / 1024 / 1024
        self.row_num = self.data.shape[0]
        self.col_num = len(self.data.columns)

        # parse columns
        self.parse_columns()
        L.info(f"build finished: {self}")
    
    def parse_columns(self):
        self.columns = OrderedDict([(col, Column(col, self.data[col])) for col in self.data.columns])

    def __repr__(self):
        return f"Table {self.name} ({self.row_num} rows, {self.data_size_mb:.2f}MB, columns:\n{os.linesep.join([repr(c) for c in self.columns.values()])})"

    def get_minmax_dict(self):
        minmax_dict = {}
        for i, col in enumerate(self.columns.values()):
            minmax_dict[i] = (col.minval, col.maxval)
        return minmax_dict
    
    def change_table_order(self,new_order):
        if new_order==None:
            return 
        # 遍历新顺序，将原始字典中的键值对按新顺序添加到空字典中
        order_list = []
        for i, key in enumerate(new_order):
            original_key = self.data.keys()[key]# 获取序号 对应的列名
            order_list.append(original_key) # 排序后的list 增加列名
        self.data =  self.data[order_list]
        self.parse_columns()
        L.info(f"ReOrder build finished: {self}")
        return

    def get_minmax_dict(self):
        minmax_dict = {}
        for i, col in enumerate(self.columns.values()):
            minmax_dict[i] = (col.minval, col.maxval)
        return minmax_dict

    def normalize(self, scale=1):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            data[cname] = col.normalize(data[cname].values) * scale
        return data

    def digitalize(self)->DataFrame:
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            if is_categorical(col.dtype):
                data[cname] = col.discretize(data[cname])
            elif col.has_nan:
                data[cname].fillna(0, inplace=True)
        # print(type(data))
        return data
#计算表列熵函数
    def set_entropy_dict(self):
        data =  self.digitalize()
        entropy_dict = {}
        for c in self.columns.keys():
            e = entropy(self.data[c].value_counts())
            entropy_dict[c] = e
        def normalize_entropy(d):
            # 计算熵值总和
            total_entropy = sum(d.values())
            
            # 归一化字典中每个键的熵值
            normalized_entropy_dict = {}
            for key, value in d.items():
                normalized_entropy_dict[key] = value / total_entropy
            
            return normalized_entropy_dict
        self.entropy_dict = normalize_entropy(entropy_dict)

    def get_original_order(self)->Tuple[List[str],List[int],str]:
        info = "\n原始顺序进行排序"
        print(info)
        order = []
        entropy_dict = {}
        order = self.columns.keys()
        order = list(order)
        
        return order, [self.data.columns.get_loc(c) for c in order], info
    
    def get_random_order(self)->Tuple[List[str],List[int],str]:
        info ="\n随机顺序进行排序" 
        print(info)
        order = []
        order = self.columns.keys()
        order = list(order)
        random.shuffle(order)
        return order, [self.data.columns.get_loc(c) for c in order],info
    
    def get_min_max_entropy_order(self):
        print("按列的熵进行排序 先是从小到大 再是从大到小")
        info = "\n按列的熵进行排序 先是从小到大 再是从大到小"
        def dict_val(x):
            return x[1]
        order = []
        self.set_entropy_dict()
        entropy_dict = self.entropy_dict

        order = dict(sorted(entropy_dict.items(),key=dict_val, reverse=False))   #False 升序 从小到大  
        entropy_dict = order
        order = order.keys()
        order1 = list(order)
        order1.reverse()
        return order, [self.data.columns.get_loc(c) for c in order] , order1 , [self.data.columns.get_loc(c) for c in order1] ,entropy_dict,  info

    def get_entropy_kmeans_group(self,method = 'pearson'):
        info = "\n按列的熵进行排序 分组"
        def dict_val(x):
            return x[1]
        order = []
        data =  self.digitalize()
        self.set_entropy_dict()
        entropy_dict = self.entropy_dict
        entropy_dict = {}

        # order = dict(sorted(entropy_dict.items(),key=dict_val, reverse=False))   #False 升序 从小到大  
        cor = data.corr(method=method)

        group =  kmeans_grouping(entropy_dict)

        L.warning("This don't finish")
        return



    def get_max_entropy_order(self):
        def dict_val(x):
            return x[1]
        order = []
        entropy_dict = {}
        for c in self.columns.keys():
            e = entropy(self.data[c].value_counts())
            entropy_dict[c] = e
        order = dict(sorted(entropy_dict.items(),key=dict_val, reverse=True))   #False 升序 从小到大  
        order = order.keys()

        return order, [self.data.columns.get_loc(c) for c in order]

    def get_min_muteinfo_order(self):

        order = []
        # find the first column with maximum entropy
        min_entropy = float('inf')
        first_col = None
        for c in self.columns.keys():
            e = entropy(self.data[c].value_counts())
            if e < min_entropy:
                first_col = c
                min_entropy = e
        assert first_col is not None, (first_col, min_entropy)
        order.append(first_col)
        sep = '|'
        chosen_data = self.data[first_col].astype(str) + sep

        # add the rest columns one by one by choosing the max mutual information with existing columns
        while len(order) < self.col_num:
            max_muinfo = float('-inf')
            next_col = None
            for c in self.columns.keys():
                if c in order: continue
                m = mutual_info_score(chosen_data, self.data[c])
                if m > max_muinfo:
                    next_col = c
                    max_muinfo = m
            assert next_col is not None, (next_col, min_entropy)
            order.append(next_col)
            # concate new chosen columns
            chosen_data = chosen_data + sep + self.data[next_col].astype(str)

        return order, [self.data.columns.get_loc(c) for c in order]
    
    def get_max_muteinfo_order(self):
        order = []

        # find the first column with maximum entropy
        max_entropy = float('-inf')
        first_col = None
        for c in self.columns.keys():
            e = entropy(self.data[c].value_counts())
            if e > max_entropy:
                first_col = c
                max_entropy = e
        assert first_col is not None, (first_col, max_entropy)
        order.append(first_col)
        sep = '|'
        chosen_data = self.data[first_col].astype(str) + sep

        # add the rest columns one by one by choosing the max mutual information with existing columns
        while len(order) < self.col_num:
            max_muinfo = float('-inf')
            next_col = None
            for c in self.columns.keys():
                if c in order: continue
                m = mutual_info_score(chosen_data, self.data[c])
                if m > max_muinfo:
                    next_col = c
                    max_muinfo = m
            assert next_col is not None, (next_col, max_entropy)
            order.append(next_col)
            # concate new chosen columns
            chosen_data = chosen_data + sep + self.data[next_col].astype(str)

        return order, [self.data.columns.get_loc(c) for c in order]

    def get_muteinfo(self, digital_data=None):
        data = digital_data if digital_data is not None else self.digitalize()
        muteinfo_dict = {}
        for c1 in self.columns.keys():
            muteinfo_dict[c1] = {}
            for c2 in self.columns.keys():
                if c1 != c2 and c2 in muteinfo_dict:
                    assert c1 in muteinfo_dict[c2], muteinfo_dict.keys()
                    muteinfo_dict[c1][c2] = muteinfo_dict[c2][c1]
                else:
                    muteinfo_dict[c1][c2] = mutual_info_score(data[c1], data[c2])
        return pd.DataFrame().from_dict(muteinfo_dict)

    def set_table_cor_ma(self,method = 'pearson'):
        data =  self.digitalize()
        cor = data.corr(method=method)
        self.cor_mate = cor

    def get_corr_sumOrder(self,method = 'pearson',order_reverse = False)->Tuple[DataFrame,List[str],List[int],str]:
        print("按每列的相关性总和排序")
        if order_reverse==False:
            info = "\n按每列的相关性总和排序" + ":从小到大"
            print("从小到大")
        else:
            info = "\n按每列的相关性总和排序" + ":从大到小"
            print("从大到小")
        data =  self.digitalize()
        cor = data.corr(method=method)
        cor_sum_dict = {}
        for key in data:
            cor_sum_dict[key] = 0
            for num in cor[key]:
                cor_sum_dict[key] += abs(num)
        def dict_val(x):
            return abs(x[1])
        
        order = dict(sorted(cor_sum_dict.items(),key=dict_val, reverse=order_reverse))   #False 升序 从小到大  
        print(order)
        order = order.keys()
        order_num_list =  [table.data.columns.get_loc(c) for c in order]
        return cor,order,order_num_list,info

    def get_corrOrder_one_by_one(self,method = 'pearson',order_reverse = False,first_col_id = 0)->Tuple[DataFrame,List[str],List[int],str]:
        print("逐列逐列的相关性排序")
        info = "\n逐列逐列的相关性排序"
        data =  self.digitalize()
        cor = data.corr(method=method)
        cor_str = cor.to_string()

        first_col_name = self.data.keys()[first_col_id]
        order =[first_col_name]#默认选了
        def dict_val(x):
            if x[1]!=1 and x[0] not in order:
                return abs(x[1])
            else:
                return -2
        for i in range(1,table.col_num):
            d = dict(cor[order[i-1]])
            k = max(d.items(),key=dict_val)
            order.append(k[0])
        # cor = dict(cor['age'])
        order_num_list =  [table.data.columns.get_loc(c) for c in order]
        return cor,order,order_num_list,info

    def get_corrOrder_one_by_one_without_relation(self,method = 'pearson',order_reverse = False,first_col_id = 0)->Tuple[DataFrame,List[str],List[int],str]:
        info = "\n逐列逐列的以最不相关性排序"
        print(info)
        data =  self.digitalize()
        cor = data.corr(method=method)
        cor_str = cor.to_string()

        first_col_name = self.data.keys()[first_col_id]
        order =[first_col_name]#默认选了
        def dict_val(x):
            if x[1]!=1 and x[0] not in order:
                return abs(x[1])
            else:
                return 2
        for i in range(1,table.col_num):
            d = dict(cor[order[i-1]])
            k = min(d.items(),key=dict_val)
            order.append(k[0])
        # cor = dict(cor['age'])
        order_num_list =  [table.data.columns.get_loc(c) for c in order]
        return cor,order,order_num_list,info
    import pandas as pd

    def find_target_col(self,col_name, corr_matrix, selected_cols,weight)->str:
        # 从字典中取出给定列名的熵值
        entropy = self.entropy_dict.get(col_name)

        # 找到与给定列相关性最大的列
        corr_series = corr_matrix[col_name].abs()
        # corr_col = corr_series.idxmax()

        # 遍历字典中的键值对，找到与给定列熵差值最小 且相关性较大的列 的列
        min_diff_col = None
        min_diff = float('inf')
        for k, v in entropy_dict.items():
            if k != col_name and k not in selected_cols:
                diff = - abs(v - entropy)*(1-weight) - abs(corr_series[k])*weight
                if diff < min_diff:
                    min_diff = diff
                    min_diff_col = k

        return min_diff_col

    def get_Hyper_Order_one_by_one(self,method = 'pearson',order_reverse = False,first_col_id = 0)->Tuple[DataFrame,List[str],List[int],str]:
        w = 0
        info = "\n逐列逐列的按熵-相关性排序 相关性系数为:{}".format(w)
        print(info)
        self.set_entropy_dict()
        entropy_dict = self.entropy_dict
        data =  self.digitalize()
        cor = data.corr(method=method)
        first_col_name = self.data.keys()[first_col_id]
        order =[first_col_name]#默认选了
        for i in range(1,table.col_num):
            # d = dict(cor[order[i-1]])
            k = self.find_target_col(order[len(order)-1],cor,order,w)
            order.append(k)
        # cor = dict(cor['age'])
        order_num_list =  [table.data.columns.get_loc(c) for c in order]
        return cor,order,order_num_list,info


    def get_corrOrder_by_first_corr(self,method = 'pearson',order_reverse=False,first_col_name = False)->Tuple[DataFrame,List[str],List[int],str]:
        if first_col_name == False:
            info = "\n和第一列有关的相关性排序"
        else:
            info = "\n和{}列有关的相关性排序".format(first_col_name)
        print(info)
        data =  self.digitalize()
        cor = data.corr(method=method)
        def dict_val(x):
            if x[1]!=1 and x[0] not in order:
                return abs(x[1])
            else:
                return -2
        if first_col_name == False:
            first_col_name = self.data.keys()[0]
        elif type(first_col_name) == int:  #input is int——column number
            first_col_name = self.data.keys()[first_col_name]
        
        cor_col_first = dict(cor[first_col_name])
        order = []
        order = dict(sorted(cor_col_first.items(),key=dict_val, reverse=order_reverse))   #False 升序 从小到大  
        order = order.keys()
        order_num_list =  [table.data.columns.get_loc(c) for c in order]
        return cor,order,order_num_list,info


    

def dump_table(table: Table) -> None:
    with open(DATA_ROOT / table.dataset / f"{table.version}.table.pkl", 'wb') as f:
        pickle.dump(table, f, protocol=PKL_PROTO)

def load_table(dataset: str, version: str, overwrite: bool=False) -> Table:
    table_path = DATA_ROOT / dataset / f"{version}.table.pkl"

    if not overwrite and table_path.is_file():
        L.info("table exists, load...")
        with open(table_path, 'rb') as f:
            table = pickle.load(f)
        L.info(f"load finished: {table}")
        return table

    table = Table(dataset, version)
    L.info("dump table to disk...")
    dump_table(table)
    return table

def dump_table_to_num(dataset: str, version: str) -> None:
    table = load_table(dataset, version)
    num_data = table.digitalize()
    csv_path = DATA_ROOT / dataset / f"{version}_num.csv"
    L.info(f"dump csv file to {csv_path}")
    num_data.to_csv(csv_path, index=False)

def print_order(order: List[int])->str:
    s = "["
    print('[',end='')
    if (len(order))==0:
        L.warning("The Order List is empty which has error!")
        return
    for o in order[:-1]:
        print(o,end=",")
        s += str(o)+","
    print(order[-1],end="]\n")
    s += str(order[-1])+"]"
    return s
    


# if __name__ == '__main__':

#     #  table = load_table('forest')
#     #  print(table.get_max_muteinfo_order())
#     # 7 1 8 6 5 9 0 4 3 2

#     #  table = load_table('census')
#     #  print(table.get_max_muteinfo_order())
#     # 4 3 2 0 6 12 7 5 1 13 9 10 8 11
#     # Kendall  相关性分析 spearman  

#     # cor_method = 'kendall'
#     cor_method = 'spearman'

#     # dataset = "dmv11"
#     # dataset = "power7"
#     # dataset = "flights_5m"
#     # dataset = "forest10"
#     # dataset = 'double_census13'
#     # dataset = 'Hyper3'
#     dataset = 'census6'
#     # dataset = 'Census13x3'
#     res_txt_path = DATA_ANALYZE_ROOT/ dataset / "analyze_res.log"
#     table = Table(dataset, 'original')
#     print(table)
#     write_info2txt(res_txt_path,"The datasets is "+dataset,extra_print=True)
#     write_info2txt(res_txt_path,"Cor method="+cor_method,extra_print=True)
#     write_info2txt(res_txt_path,"DEBUG",extra_print=True)

#     #原始排序
#     order,order_num_list,info = table.get_original_order()
#     ors = print_order(order=order_num_list)
#     write_info2txt(res_txt_path,info,extra_print=True)
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名

#     #随机排序
#     order,order_num_list,info = table.get_random_order()
#     ors = print_order(order=order_num_list)
#     write_info2txt(res_txt_path,info,extra_print=True)
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名
#     #随机排序结束

#     #按熵排序
#     order,order_num_list,order1,order_num_list1,entropy_dict,info = table.get_min_max_entropy_order()
#     write_info2txt(res_txt_path,info,True)
#     write_info2txt(res_txt_path,entropy_dict,True)
#     write_info2txt(res_txt_path,order,True)
#     write_info2txt(res_txt_path,order_num_list,True)
#     write_info2txt(res_txt_path,order1,True)
#     write_info2txt(res_txt_path,order_num_list1,True)
#     #按熵排序结束
#     write_info2txt(res_txt_path,cor_method,extra_print=True)
#     # one by one 排序 下一列找当前列关系值最大的列
#     begin_col_num = 0
#     cor,order,order_num_list,info = table.get_corrOrder_one_by_one(method=cor_method,order_reverse=False,first_col_id=begin_col_num)
#     cor_str = cor.to_string()
#     order_num_list =  [table.data.columns.get_loc(c) for c in order]
#     print(order_num_list)# 编号
#     write_info2txt(res_txt_path,info)
#     write_info2txt(res_txt_path,cor_str,True)
#     ors = print_order(order=order_num_list)#紧密的输出
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名
#     # one by one相关性排序结束

#     # one by one 最不相关性排序
#     begin_col_num = 0
#     cor,order,order_num_list,info = table.get_corrOrder_one_by_one_without_relation(method=cor_method,order_reverse=False,first_col_id=begin_col_num)
#     cor_str = cor.to_string()
#     order_num_list =  [table.data.columns.get_loc(c) for c in order]
#     print(order_num_list)# 编号
#     write_info2txt(res_txt_path,info)
#     write_info2txt(res_txt_path,cor_str,True)
#     ors = print_order(order=order_num_list)#紧密的输出
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名

#     # 按与第一列 或 指定列 相关的相关性排序
#     write_info2txt(res_txt_path,cor_method,extra_print=True)
#     col_num = 0
#     cor,order,order_num_list,info = table.get_corrOrder_by_first_corr(method=cor_method,order_reverse=False,first_col_name = col_num)
#     cor_str = cor.to_string()
#     order_num_list =  [table.data.columns.get_loc(c) for c in order]
#     print(order_num_list)# 编号
#     write_info2txt(res_txt_path,info)
#     # write_info2txt(res_txt_path,cor_str,True)
#     ors = print_order(order=order_num_list)#紧密的输出
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名
#     #每一列的与其他列的相关性总和排序
#     write_info2txt(res_txt_path,cor_method,extra_print=True)
#     cor,order,order_num_list,info = table.get_corr_sumOrder(method=cor_method,order_reverse=False)
#     cor_str = cor.to_string()
#     order_num_list =  [table.data.columns.get_loc(c) for c in order]
#     relation_order_list = order_num_list
#     print(order_num_list)# 编号
#     write_info2txt(res_txt_path,info)
#     # write_info2txt(res_txt_path,cor_str,True)
#     ors = print_order(order=order_num_list)#紧密的输出
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名
#     # #相关性排序结束


#     # one by one  混合 熵的差 与相关性进行排序
#     write_info2txt(res_txt_path,"以相关性最大的列为首",True)
#     begin_col_num = relation_order_list[-1]
#     cor,order,order_num_list,info = table.get_Hyper_Order_one_by_one(method=cor_method,order_reverse=False,first_col_id=begin_col_num)
#     cor_str = cor.to_string()
#     order_num_list =  [table.data.columns.get_loc(c) for c in order]
#     print(order_num_list)# 编号
#     write_info2txt(res_txt_path,info)
#     write_info2txt(res_txt_path,cor_str,True)
#     ors = print_order(order=order_num_list)#紧密的输出
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名
#     # one by one相关性排序结束

#     # one by one  混合 熵的差 与相关性进行排序
#     begin_col_num = 0
#     cor,order,order_num_list,info = table.get_Hyper_Order_one_by_one(method=cor_method,order_reverse=False,first_col_id=begin_col_num)
#     cor_str = cor.to_string()
#     order_num_list =  [table.data.columns.get_loc(c) for c in order]
#     print(order_num_list)# 编号
#     write_info2txt(res_txt_path,info)
#     write_info2txt(res_txt_path,cor_str,True)
#     ors = print_order(order=order_num_list)#紧密的输出
#     write_info2txt(res_txt_path,ors,extra_print=True) # 显示顺序编号
#     write_info2txt(res_txt_path,order,extra_print=True) # 显示实际的列名
#     # one by one相关性排序结束

#     # 4 0 1 2 3 5 8 7 6
