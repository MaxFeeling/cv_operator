from algo_cli.params import DataSource,WorkDir
from algo_cli import Logger, args, operator,io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import argparse, torch, os
from io import BytesIO

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
p_logger = Logger()

@operator("cls_eval",output="Table",domain="default",display_name="模型评估-图像分类",doc_url="test",scenario="通用分类")
def main(args):
    """
    对图像分类模型的效果进行评估，了解更多图像分类模型评估 指标点击这里
    """
    dfs = io.read_table(args.inputs)
    ground_truth =dfs.values[:,0].astype(np.int32)
    prediction = dfs.values[:,1].astype(np.int32)
    eval = {'acc': [],'P':[],'R':[],'F1':[]}
    acc = accuracy_score(ground_truth,prediction)
    eval['acc'].append(acc*100)
    precision = precision_score(ground_truth,prediction,average='macro')
    eval['P'].append(precision*100)
    recall = recall_score(ground_truth,prediction,average='macro')
    eval['R'].append(recall*100)
    f1 = f1_score(ground_truth,prediction,average='macro')
    eval['F1'].append(f1*100)
    df = pd.DataFrame(eval)
    df.to_csv(os.path.join(args.model_dir, "eval_results.csv"), index=False)
    p_logger.table("eval_result", os.path.join(args.model_dir, "eval_results.csv"))

@args("cls_eval")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=DataSource("Table"),
                        default="./models/predictions.csv",
                        help='输入的数据', required=True)
    parser.add_argument("--model_dir", type=WorkDir, default='./models', help='保存模型路径', required=False)
    return parser