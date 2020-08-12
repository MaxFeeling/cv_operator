# -*- coding: utf-8 -*-
from algo_cli.params import DataSource,WorkDir
from algo_cli import Logger, args, operator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import argparse, torch, os
from ResNet.data import *
from io import BytesIO
import csv
p_logger = Logger()

@operator("resnet_test",output="Table",domain="default",display_name="resnet模型预测",doc_url="test",scenario="通用分类")
def predict(args):
    """
    使用resnet模型对新的图片数据进行预测，得到预测结果，了解resnet
    """
    model_path, imageset_path = args.inputs.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_paths, label_path = process_imageset(imageset_path)
    model = torch.load(model_path)#加载模型
    data = img_paths
    model = model.to(device).eval()
    pred_dataset = ImageData(data,label=label_path,train_mode='val')
    pred_loader = DataLoader(dataset=pred_dataset, batch_size=64, num_workers=4)

    pred_dict = {'ground_truth':[],'pred': [], 'confidence': []}
    header = ['ground_truth','pred','confidence']
    total,correct = 0,0
    with torch.no_grad():
        for batch_idx, test in enumerate(pred_loader):
            inputs,labels = test
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            percentage = torch.nn.functional.softmax(outputs.data, dim=1)

            for j in range(len(inputs)):
                pred_index = predicted[j].item()
                label = labels[j].item()
                prob = percentage[j][pred_index].item()
                pred_dict['pred'].append(pred_index)
                pred_dict['ground_truth'].append(label)
                pred_dict['confidence'].append(prob)
                total+=1
                if pred_index==labels[j]:
                    correct+=1
                print('labels_pred: %s prob: %.3f' % (pred_index, prob))
    acc = 100 * correct / total
    print('Accuracy of the network on the %d test images: %d %%' % (total, acc))
    df = pd.DataFrame(pred_dict)
    df.to_csv(os.path.join(args.model_dir, "predictions.csv"), header=header,index=False)
    p_logger.table("prediction_result", os.path.join(args.model_dir, "predictions.csv"))


@args("resnet_test")
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputs', type=DataSource("Model,ImageGroup"),
                        default="./saved_models/saved_model,./mnist200/labels.csv",
                        help='输入的数据', required=True)
    parser.add_argument("--model_dir", type=WorkDir, default='./models', help='保存模型路径', required=True)
    return parser

