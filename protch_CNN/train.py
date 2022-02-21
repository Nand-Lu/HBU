# Import需要的套件
import sys
import os
import json
# from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.CNN import CNN
from model.resnet import *
from hyper_params  import  hyper_params
import numpy as np
from prettytable import PrettyTable
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from torchtoolbox.transform import Cutout
import random


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        file_handle = open(r'.\Result\result.txt', mode='a')
        print("the model accuracy is ", acc,file=file_handle)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

        file_handle.write(str(table))
        file_handle.close()


    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.figure(2)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()

        plt.savefig("confusion_matrix.jpg")
        # plt.show()

def set_seed(seed):#设置随机种子，保证结果的可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_root = os.path.abspath(os.path.join(os.getcwd(),"datasets"))  # get data root path
    print(data_root)

    set_seed(20)
    #//////////////设置数据集路径\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    image_path = os.path.join(data_root, "dogs_cats","data")
    print(image_path)

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    data_transform = {
            "train": transforms.Compose([# transforms.CenterCrop(224),单通道灰度图需要这行
                                         transforms.Resize((hyper_params["input_size"],hyper_params["input_size"])),#transforms.RandomResizedCrop(224),
                                         #Cutout(),效果不见得会好，与下面的图像增强最好不一起用，慎用
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(p=1),
                                         transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0),
                                         transforms.RandomRotation(60, resample=False, expand=False,
                                                                               center=None, fill=None),
                                         # transforms.RandomCrop(224, padding=None, pad_if_needed=False, fill=0, padding_mode='edge'),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         # transforms.Normalize([0.485, ], [0.229,])单通道灰度图需要这行
                                         ]),
            "val": transforms.Compose([
                                         # transforms.CenterCrop(224),单通道灰度图需要这行
                                       transforms.Resize((hyper_params["input_size"],hyper_params["input_size"])),

                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       # transforms.Normalize([0.485, ], [0.229, ])单通道灰度图需要这行
                                       ])}


    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    data_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in data_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    nw = min([os.cpu_count(), hyper_params['batch_size'] if hyper_params['batch_size'] > 1 else 0, 8])

    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size= hyper_params['batch_size'], shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "valid"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=hyper_params['batch_size'], shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    # ////////////////////选择网络模型///////////////////////////////////////////////////////////////////////////////////////////////////
    net = resnet101()
    # ///////////预训练/////////////

    model_dict = net.state_dict()

    pretrained_dict = torch.load(hyper_params["pre_modelpath"])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    net.to(device)
    # //////////////////////////////////
    # define loss function


    loss_function = nn.CrossEntropyLoss()



    train_loss=[]
    val_accuracy=[]


    # construct an optimizer


    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr= hyper_params['learning_rate'])

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # ////////////////学习率衰减/////////////
    lr_list = []

    epoch_path=0

    best_acc = 0.0
    # save_path = './CNN_best.pth'
    train_steps = len(train_loader)

    
    for epoch in range( hyper_params['epochs']):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        for step, data in enumerate(train_bar):
            images, labels = data


            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()


            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     hyper_params["epochs"],
                                                                     loss)
            #..................................................................................

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           hyper_params["epochs"])

        val_accurate = acc / val_num
#................................log..................................................................
        log_save = open(r'.\log\log_cutout.txt', mode='a')

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f ' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f ' %
              (epoch + 1, running_loss / train_steps, val_accurate),file=log_save)
        # print(print_loss)
        # print(print_loss,file=log_save)
        log_save.close()

        train_loss.append(running_loss / train_steps)
        val_accuracy.append(val_accurate)
        x=range(0,hyper_params["epochs"])
        epoch_path+=1


        if epoch_path%5==0:#5个epoch保存一次
            save_path1=os.path.join(save_path , str(epoch_path)+"epoch.pth")
            torch.save(net.state_dict(), save_path1)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), pth_path)
        scheduler.step()

        #---------画图---------
    plt.figure(1)
    plt.plot(x, train_loss, label='train_loss', linewidth=2, color='r', marker='o',
             markerfacecolor='blue', markersize=2)
    plt.plot(x, val_accuracy, label='val_accurary')
    plt.xlabel('epoch')
    plt.ylabel('rate')
    plt.title('loss')
    plt.legend()
    plt.savefig('loss.jpg')
    # plt.show()

    plt.figure(3)
    plt.plot(range(hyper_params["epochs"]) , lr_list, color='r')
    plt.savefig('lr.jpg')
   # print(best_acc is )
    print('Finished Training')
# /////////////////////////混淆矩阵/////////////////////////////////////
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()



if __name__ == '__main__':
    # 生成最佳参数的保存路径
    save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.abspath(__file__))),"Result")
    pth_path = os.path.join(save_path,"best_"+"CNNpth")
    main()
