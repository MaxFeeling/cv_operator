#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
from algo_cli.params import DataSource,WorkDir
from algo_cli import Logger, args, operator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import argparse, torch, os
from io import BytesIO
from torchvision import transforms

p_logger = Logger()


class ToolDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = np.array(image)
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        img = Image.open(image)
        img = img.convert('RGB')
        # img = img.resize((224, 224))
        # img = np.asarray(img, dtype=np.float32)

        if self.transform:
            img = self.transform(img)

        return img


@operator("resnet_predictor", runner="default_http_server", output="Table")
class predictor:
    def __init__(self, args):
        #加载模型
        model = torch.load(args.model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
    def predict(self,data,query_args):  # data为在线请求数据，query_args为在线请求url的参数

        base_dir = ""
        image_file = os.path.join(base_dir,data)
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),])
        images = os.listdir(image_file)
        images = list(map(lambda image:os.path.join(image_file,image),images))
        pred_dataset = ToolDataset(images,transform=transform)
        pred_loader = DataLoader(dataset=pred_dataset, batch_size=1, num_workers=0)
        pred_dict = {'label_pred': [], 'prob': []}
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pred_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                percentage = torch.nn.functional.softmax(outputs, dim=1)
                for j in range(len(inputs)):
                    pred_index = predicted[j].item()
                    prob = percentage[j][pred_index].item()
                    pred_dict['label_pred'].append(pred_index)
                    pred_dict['prob'].append(prob)
                    print('labels_pred: %s acc: %.3f' % (pred_index, prob))
        print('Predict successfully')
        return pred_dict


@args("resnet_predictor")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=DataSource("Model"), required=True, help='加载模型的路径',default="./models/saved_model")
    return parser



# @operator("resnet_predictor", runner="default_http_server", output="Table")
# class predictor:
#     def __init__(self, args):
#         #加载模型
#         model = torch.load(args.model_dir)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = model.to(self.device).eval()
#     def predict(self,data,query_args):  # data为在线请求数据，query_args为在线请求url的参数
#         #加载图片
#         image_file = data
#         images = os.listdir(image_file)
#         images = list(map(lambda image: os.path.join(image_file, image), images))
#         transform = transforms.Compose([
#             transforms.Resize(224),
#             transforms.ToTensor(), ])
#         images = os.listdir(image_file)
#         images = list(map(lambda image: os.path.join(image_file, image), images))
#         pred_dataset = ToolDataset(images, transform=transform)
#         pred_loader = DataLoader(dataset=pred_dataset, batch_size=1, num_workers=0)
#
#         #预测流程
#         pred_dict = {'label_pred': [], 'prob': []}
#         with torch.no_grad():
#             for batch_idx, inputs in enumerate(pred_loader):
#                 inputs = inputs.to(self.device)
#                 outputs = self.model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 percentage = torch.nn.functional.softmax(outputs, dim=1)
#                 for j in range(len(inputs)):
#                     pred_index = predicted[j].item()
#                     prob = percentage[j][pred_index].item()
#                     pred_dict['label_pred'].append(pred_index)
#                     pred_dict['prob'].append(prob)
#                     print('labels_pred: %s acc: %.3f' % (pred_index, prob))
#         print('Predict successfully')
#         return pred_dict
#
#
# """
#         在线预估的方式是向服务器发送post请求
#         如果post文件:curl -F "data=@D:/minist200/01.jpg" ”http://xxx.xxx.xxx.xxx“
#         return:data=file(file:"01.jpg",body:"01.jpg"的二进制文件)
#         如果postjson:curl -H "Content-Type: application/json" -X POST '{"data":"D:/minist200/"}' http://xxx.xxx.xxx.xxx
#         return:data="D:/minist200/"
#         """
#
#
