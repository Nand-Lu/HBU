import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from model.resnet import *

# 图片预处理
def img_preprocess(img_in):
    img = img_in.copy()
    img = cv2.resize(img,(128,128))
    img = img[:, :, ::-1]   				# 1
    img = np.ascontiguousarray(img)			# 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img
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

    path_cam_img = os.path.join(out_dir, "cam_"+img_name)
    cv2.imwrite(path_cam_img, cam_img)


if __name__ == '__main__':
    img_name = "007-7233-400.jpg"

    img_path = r"C:\Users\HZY_PC\Desktop\DR\test\1"+"/"+img_name

    pthfile = r"C:\Users\HZY_PC\Desktop\DR\100epoch_lrdown_迁移学习\100epoch.pth"
    json_path = './class_indices.json'
    output_dir = './Result/cam'

    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value)
               in load_json.items()}

    # 只取标签名
    classes = list(classes.get(key) for key in range(2))

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(img_path, 1)
    img_input = img_preprocess(img)

    # 加载 squeezenet1_1 预训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = resnet50().to(device)

    net.load_state_dict(torch.load(pthfile))
    net.eval()  # 8
    # print(net)

    # 注册hook
    net.layer4[2].conv3.register_forward_hook(farward_hook)
    net.layer4[2].conv3.register_full_backward_hook(backward_hook)
    # net.features[-1].expand3x3.register_forward_hook(farward_hook)  # 9
    # net.features[-1].expand3x3.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img, fmap, grads_val, output_dir)
