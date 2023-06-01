"""Le Carb - LEarned CARdinality estimator Benchmark

Usage:
  lecarb train_seed [-s <seed>] [-d <dataset>] [-w <workload>]
  lecarb train_order [-s <seed>] [-d <dataset>] [-w <workload>] [-f <final_col>]
  lecarb analyze [-d <dataset>] [-w <workload>] [-t <type>] 
  lecarb train [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [-e <estimator>] [--params <params>] [--sizelimit <sizelimit>]

Options:
  -s, --seed <seed>                Random seed.
  -t, --type <type>                Analyze type.
  -f, --final_col <final_col>      Final_col.
  -d, --dataset <dataset>          The input dataset [default: census13].
  -v, --dataset-version <version>  Dataset version [default: original].
  -w, --workload <workload>        Name of the workload [default: base].
  -e, --estimator <estimator>      Name of the estimator [default: naru].
  --params <params>                Parameters that are needed.
  --sizelimit <sizelimit>          Size budget of method, percentage to data size [default: 0.015].
  --no-label                       Do not generate ground truth label when generate workload.
  --overwrite                      Overwrite the result.
  -o, --old-version <old_version>  When data updates, query should focus more on the new data. The <old version> is what QueryGenerator refers to.
  -r, --win-ratio <ratio>          QueryGen only touch last <win_ratio> * size_of(<old version>).
  --sample-ratio <sample-ratio>    Update query set with sample of dataset
  -h, --help                       Show this screen.
"""
from ast import literal_eval
from time import time
import json
from docopt import docopt
import os
import itertools
import logging
import math
# import jsonlines
import csv
L = logging.getLogger(__name__)
from .workload.gen_workload import generate_workload
from .workload.gen_label import generate_labels, update_labels
from .workload.merge_workload import merge_workload
from .workload.dump_quicksel import dump_quicksel_query_files, generate_quicksel_permanent_assertions
from .dataset.dataset import load_table, dump_table_to_num,Table
from .dataset.gen_dataset import generate_dataset
from .dataset.manipulate_dataset import gen_appended_dataset
from .estimator.sample import test_sample
from .estimator.postgres import test_postgres
from .estimator.mysql import test_mysql
from .estimator.mhist import test_mhist
from .estimator.bayesnet import test_bayesnet
from .estimator.feedback_kde import test_kde
from .estimator.utils import report_errors, report_dynamic_errors
from .estimator.naru.naru import train_naru, test_naru, update_naru
from .estimator.mscn.mscn import train_mscn, test_mscn
from .estimator.lw.lw_nn import train_lw_nn, test_lw_nn
from .estimator.lw.lw_tree import train_lw_tree, test_lw_tree
from .estimator.deepdb.deepdb import train_deepdb, test_deepdb, update_deepdb
from .workload.workload import dump_sqls
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def write_list2csv(result_file,List,row_head):
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row_head)
        for l in List:
            writer.writerow(l)

def get_order_id(order:str,val:int):
    #val : 列数
    final_col  = int(order[-1])
    ori_order = list(range(val))
    ori_order.remove(final_col)
    part = math.factorial(val-1)
    i = -1 + final_col * part
    length = 0
    for perm in itertools.permutations(ori_order, len(ori_order)):
        i += 1
        if i < length:
            continue
        now_order = list(perm)+[final_col]
        print(now_order)
        now_order_str = ",".join(map(now_order))
        if now_order_str == order:
            return i
    return -1


if __name__ == "__main__":
    args = docopt(__doc__, version="Le Carb 0.1")

    # print("helloe")
    if args["train_seed"]:
        dataset = args["--dataset"]
        workload = args["--workload"]
        # params = literal_eval(args["--params"])
        params = {'epochs': 100 , 'input_encoding': 'embed', 'output_encoding': 'embed', \
        'embed_size': 8, 'layers': 4, 'fc_hiddens': 21, 'residual': True, 'warmups': 0,"order":[0,1,2,3,4,5,6,7]}
        sizelimit = 0.15
        all_time = 0
        result_dict = {}
        mean_error_list = []
        th95_error_list = []

        for seed in range(0,100):
            start_time = time()
            result_dict = train_naru(seed, dataset, 'original', workload, params, sizelimit, result_dict=result_dict, GPU_id= 1)
                # infomation_dict = {"seed":seed,"order":order_name,"error":{workload: metrics},
                    #    "model_name":f"{table.version}-{model.name()}_warm{args.warmups}-{seed}"}
            mean_error_list.append(result_dict["error"][workload]['mean'])
            th95_error_list.append(result_dict["error"][workload]['95th'])
            if os.path.isdir(f"myresult/{dataset}/{workload}")== False:
                os.makedirs(f"myresult/{dataset}/{workload}")
            with open(f"myresult/{dataset}/{workload}/my_dict.json", "a+") as f:
                json.dump(result_dict, f)
                f.write('\n')
            end_time = time()
            use_time = (end_time - start_time)/60
            all_time += use_time
            L.info(f"the train use {use_time} min")
            L.info(f"the all time {all_time} min")

        
    elif args["train_order"]:
        # print("______________")
        dataset = args["--dataset"]
        workload = args["--workload"]
        # final_col = args["--final_col"]
        final_col_list = literal_eval(args["--final_col"])
        print()

        # print(type(final_col))
        # exit(0)
        val = int(dataset[-1])
        print(val)
        params = {'epochs': 80 , 'input_encoding': 'embed', 'output_encoding': 'embed', \
        'embed_size': 8, 'layers': 4, 'fc_hiddens': (val-1)*3, 'residual': True, 'warmups': 0}
        sizelimit = 0.15
        all_time = 0
        result_dict = {}
        mean_error_list = []
        th95_error_list = []
        seed = 123
        
        ori_order = list(range(val))
        final_col = 0
        # ori_order.remove(first_col)
# poetry run python -m lecarb train_order --dataset census8 --workload basenormal --final_col "[6,7]"
        for final_col in final_col_list:
            ori_order = list(range(val))
            ori_order.remove(final_col)
            i = -1
            length = 0
            if os.path.exists(f"myresult/{dataset}/{workload}/my_dict_train_order_{final_col}.json"):
                with open(f"myresult/{dataset}/{workload}/my_dict_train_order_{final_col}.json", "r") as f:
                    result_dict_list = f.readlines()
                    length = len(result_dict_list)
                    f.close()
            print("the lenght is:",length)

            for perm in itertools.permutations(ori_order, len(ori_order)):
                i += 1
                if i < length:
                    continue

                now_order = list(perm)+[final_col]
                print(now_order)
                params["order"] = now_order
                start_time = time()
                result_dict.update({"id":i})
                result_dict.update(train_naru(seed, dataset, 'original', workload, params, sizelimit, result_dict=result_dict, GPU_id = "2"))
                    # infomation_dict = {"seed":seed,"order":order_name,"error":{workload: metrics},
                        #    "model_name":f"{table.version}-{model.name()}_warm{args.warmups}-{seed}"}
                mean_error_list.append(result_dict["error"][workload]['mean'])
                th95_error_list.append(result_dict["error"][workload]['95th'])
                if os.path.isdir(f"myresult/{dataset}/{workload}")== False:
                    os.makedirs(f"myresult/{dataset}/{workload}")
                with open(f"myresult/{dataset}/{workload}/my_dict_train_order_{final_col}.json", "a+") as f:
                    json.dump(result_dict, f)
                    f.write('\n')
                end_time = time()
                use_time = (end_time - start_time)/60
                all_time += use_time
                L.info(f"the train use {use_time} min")
                L.info(f"the all time {all_time} min")
            
    elif args["analyze"]:
        # the analyze test 
        # poetry run python -m lecarb analyze --dataset census6 --workload basenormal --type train_order
        import jsonlines
        print("hello")
        dataset = args["--dataset"]
        workload = args["--workload"]
        target_type = args["--type"]
        
        val = int(dataset[-1])
        json_name = "my_dict_train_order_"
            # continue finish for many file to analyze
        result_dict_list = []
        if json_name[-1] == '_':
            for i in range(val):
                print(f"myresult/{dataset}/{workload}/{json_name}{i}.json")
                with jsonlines.open(f"myresult/{dataset}/{workload}/{json_name}{i}.json", "r") as f:
                    for obj in f:
                        result_dict_list.append(obj)
                    f.close()
        else:
            with jsonlines.open(f"myresult/{dataset}/{workload}/{json_name}.json", "r") as f:
                for obj in f:
                    result_dict_list.append(obj)

        length = len(result_dict_list)
        print(result_dict_list[0])
        row_head = ['order','max','95th','90th','mean','10%','20%','30%','40%','50%','60%','70%','80%','90%']
        row_list = []
        i = 0
        order_id_dict = {}
        for res_dict in result_dict_list:
            error_m = res_dict['error'][workload]
            row_data = [res_dict['order'],error_m["max"],error_m["95th"],
                        error_m["90th"],error_m["mean"],error_m['10%'],
                        error_m['20%'],error_m['30%'],error_m['40%'],error_m['50%'],
                        error_m['60%'],error_m['70%'],error_m['80%'],error_m['90%']]
            # print(row_data[0])
            # print(type(row_data[0]))
            order_id_dict[res_dict['order']] = i
            row_list.append(row_data)
            i += 1
        model_name = result_dict_list[0]["model_name"]
        write_list2csv(f"myresult/{dataset}/{workload}/{dataset}_{workload}_{model_name}_res.csv",row_list,row_head=row_head)
            # print(res_dict['error'][workload]["max"])
            # print(type(res_dict["error"][workload]['max']))
        # dataset = 'census6'
        # 创建Order 
        test_order = '0,4,2,3,1,5,6'
        test_id = order_id_dict[test_order]
        print(row_list[test_id])

    if args["train"]:
        dataset = args["--dataset"]
        version = args["--dataset-version"]
        workload = args["--workload"]
        params = literal_eval(args["--params"])
        sizelimit = float(args["--sizelimit"])

        if args["--estimator"] == "naru":
            train_naru(seed, dataset, version, workload, params, sizelimit)
        elif args["--estimator"] == "mscn":
            train_mscn(seed, dataset, version, workload, params, sizelimit)
        elif args["--estimator"] == "deepdb":
            train_deepdb(seed, dataset, version ,workload, params, sizelimit)
        elif args["--estimator"] == "lw_nn":
            train_lw_nn(seed, dataset, version ,workload, params, sizelimit)
        elif args["--estimator"] == "lw_tree":
            train_lw_tree(seed, dataset, version ,workload, params, sizelimit)
        else:
            raise NotImplementedError
        exit(0)



        table = Table(dataset, 'original')
        table.set_entropy_dict()
        entropy_dict = table.entropy_dict
        table.set_table_cor_ma()
        




            
    # if 

    