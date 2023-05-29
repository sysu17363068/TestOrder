"""Le Carb - LEarned CARdinality estimator Benchmark

Usage:
  lecarb train_seed [-s <seed>] [-d <dataset>] [-w <workload>]
  lecarb train_order [-s <seed>] [-d <dataset>] [-w <workload>] [-f <final_col>]
  lecarb analyze [-d <dataset>] [-w <workload>] [-t <type>] 

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
L = logging.getLogger(__name__)
from .workload.gen_workload import generate_workload
from .workload.gen_label import generate_labels, update_labels
from .workload.merge_workload import merge_workload
from .workload.dump_quicksel import dump_quicksel_query_files, generate_quicksel_permanent_assertions
from .dataset.dataset import load_table, dump_table_to_num
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
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
        params = {'epochs': 80 , 'input_encoding': 'embed', 'output_encoding': 'embed', \
        'embed_size': 8, 'layers': 4, 'fc_hiddens': 18, 'residual': True, 'warmups': 0}
        sizelimit = 0.15
        all_time = 0
        result_dict = {}
        mean_error_list = []
        th95_error_list = []
        seed = 123
        val = int(dataset[-1])
        print(val)
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
        print("hello")
        dataset = args["--dataset"]
        workload = args["--workload"]
        target_type = args["--type"]
        # poetry run python -m lecarb analyze --dataset census6 --workload basenormal --type train_order
        
        with open(f"myresult/{dataset}/{workload}/my_dict_{target_type}.json", "r") as f:
            result_dict_list = json.load(f)
        length = len(result_dict_list)
        for res_dict in result_dict_list:
            print(res_dict['error'][workload])

            
    # if 

    