import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from model.resnet import *
from torchvision import models
import json
from PIL import Image
from hyper_params  import  hyper_params

# 图片预处理

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)
# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam_txt_30epoch_"+img_name)
    cv2.imwrite(path_cam_img, cam_img)


if __name__ == '__main__':
    img_name = "007-7233-400.jpg"

    img_path = r"C:\Users\HZY_PC\Desktop\DR\test\1"+"/"+img_name
    weights_path = r"./Result/30epoch.pth"
    json_path = './class_indices.json'
    output_dir = './Result/cam'

    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value)
               in load_json.items()}

    # 只取标签名
    classes = list(classes.get(key) for key in range(2))
    # print(classes)
    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            # transforms.Grayscale(1),
         transforms.Resize((hyper_params["input_size"],hyper_params["input_size"])),
         #transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         # transforms.Normalize([0.485, ], [0.229, ])
         ])

    # load image


    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)


    img = Image.open(img_path)
    img1 = cv2.imread(img_path, 1)


    # [N, C, H, W]
    img = data_transform(img)

    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model =resnet50().to(device)

    # load model weights


    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()



    model.layer4[2].conv3.register_forward_hook(farward_hook)
    model.layer4[2].conv3.register_full_backward_hook(backward_hook)



    # forward

    output = model(img)



    idx = np.argmax(output.cpu().data.numpy())

    print("predict: {}".format(classes[idx]))

    # backward
    model.zero_grad()
    class_loss = output[0, idx]

    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img1, fmap, grads_val, output_dir)
    model =resnet50().to(device)

    # load model weights


