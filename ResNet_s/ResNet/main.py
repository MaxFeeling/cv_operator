from ResNet.data import *
from ResNet.Resnets import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR,StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse, torch, os, time
from algo_cli import io, args, operator, Logger
import argparse
from algo_cli.params import DataSource, WorkDir,File,ModelPath,ModelDir,PretrainedModel
import torch.autograd as autograd
p_logger = Logger()

@args("resnet_series")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,  help='可选模型', default="resnet50",
                        metavar="模型选择",choices=["resnet18","resnet34","resnet50","resnet101","resnet152"])
    parser.add_argument("--model_dir", type=WorkDir, required=False, help='保存模型路径'
                        , default="./models")
    parser.add_argument("--pretrain_dir",type=PretrainedModel(),help="预训练模型路径",
                        required=False,default="./pretrain_models")
    parser.add_argument('--input_data_source', type=DataSource("ImageGroup"),
                        default="./mnist_png/labels.csv",help='输入的数据')
    parser.add_argument("--num-epoch", type=int, default=2, metavar="迭代轮数",
                        help="训练过程设定的迭代轮数，一个epoch表示要遍历一次数据")
    parser.add_argument("--batch_size", type=int, default=6, metavar="训练批次大小",
                        help="训练时，一个批中的样本总数")
    parser.add_argument("--learning_rate", type=float, default=0.001,metavar="学习率",
                        help="学习率初始值，学习率决定了每次梯度下降要以多大程度调整网络的权重")
    return parser

@operator("resnet_series", version="1.0.0", output="Model",domain="default",display_name="resnet模型训练",doc_url="test",scenario="通用分类")
def main(args):
    """
    使用resnet卷积神经网络训练图像分类模型，了解resnet
    """
    resnet_train(data_path=args.input_data_source, save_path=args.model_dir, batch_size=args.batch_size,epoch_num=args.num_epoch,learning_rate=args.learning_rate,pretrain_dir=args.pretrain_dir)


def resnet_train(data_path = "CACD2000/", model_name = "resnet50",batch_size=32,save_path="./models", epoch_num=1, learning_rate=0.001,loadPretrain=0,pretrain_dir="./pretrain_models/saved_model"):
    #data process
    img_paths, labels = process_imageset(data_path)
    number_classes = len(set(labels))
    x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=0.2, random_state=0)
    train_dataset = ImageData(x_train, y_train, train_mode="train")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_dataset = ImageData(x_test, y_test, train_mode="val")
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    if model_name == "resnet18":
        model = resnet18(pretrained=(loadPretrain == 1), num_classes=number_classes)
    elif model_name == "resnet34":
        model = resnet34(pretrained=(loadPretrain == 1), num_classes=number_classes)
    elif model_name == "resnet50":
        model = resnet50(pretrained=(loadPretrain == 1), num_classes=number_classes)
    elif model_name == "resnet101":
        model = resnet101(pretrained=(loadPretrain == 1), num_classes=number_classes)
    elif model_name == "resnet152":
        model = resnet152(pretrained=(loadPretrain == 1), num_classes=number_classes)
    if torch.cuda.device_count() > 1:
        print("There are ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("There is only one GPU")
    else:
        print("Only use CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(os.path.dirname(__file__))
    # print(os.path.join(os.path.dirname(__file__),"pretrain_models","saved_model"))
    # model = torch.load(os.path.join(os.path.dirname(__file__),"pretrain_models","saved_model"))
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.001)
    # lambda1 = lambda epoch: np.sin(epoch) / epoch
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda1)


    for epoch in range(epoch_num):
        p_logger.progress(round(epoch/epoch_num))
        for i_batch, sample_batch in enumerate(train_loader):
            s_time = time.time()

            # Step.1 Load data and label
            images_batch, labels_batch = sample_batch
            labels_batch = torch.LongTensor(labels_batch.view(-1).numpy())
            input_image, target_label = images_batch.to(device),labels_batch.to(device)


            # Step.2 calculate loss
            output = model(input_image)
            loss = loss_function(output, target_label)

            # Step.3 Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            print('"Epoch%d[%d]: %.5f, Time is %.2fs"' % (epoch + 1, i_batch + 1, loss.item(), time.time() - s_time))
            p_logger.scalar("train_loss",loss.item(),1)
            torch.save(model, os.path.join(save_path, 'saved_model'))
        print('Finished Training')

        # Check Result
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs,labels = inputs.to(device),labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print('Accuracy of the network on the %d test images: %d %%' % (total, acc))

        p_logger.model("pytorch_model",
                       os.path.join(save_path, "saved_model"),
                       predictor="resnet_predictor",
                       model_format="saved_model")
    p_logger.progress(1.0)



