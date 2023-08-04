import warnings
warnings.filterwarnings("ignore")

import argparse
from tqdm import tqdm
import os
import time
import copy
import random

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import quan_resnet_cifar, quan_resnet_imagenet, quan_vgg_cifar, quan_vgg_imagenet, resnet_imagenet, vgg_imagenet
from models.quantization import *
import numpy as np
import config
from models.AugLag import augLag_Linear

show_init_info = True

def initialize_env(seed: int = 42):
    print(f"pid {os.getpid()}")
    set_seed(seed)
    gpu_select()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

def gpu_select():
    exp = config.no_use_gpu_id
    import pynvml
    pynvml.nvmlInit()

    deviceCount = pynvml.nvmlDeviceGetCount()
    selected_index = 0
    min_used_ratio = 1
    for idx in range(deviceCount):
        if idx in exp or str(idx) in exp:continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_ratio = info.used / info.total
        if used_ratio < min_used_ratio:
            min_used_ratio = used_ratio
            selected_index = idx

    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_index)


def load_state_dict(net, checkpoint):
    global show_init_info
    if show_init_info:
        if 'acc' in checkpoint.keys():
            print(f"Float-point model Accuracy: {checkpoint['acc']}")
            show_init_info = False
    state_tmp = net.state_dict()

    b_ws = {}
    for key in state_tmp.keys():
        if 'b_w' in key:
            b_ws[key] = state_tmp[key]

    if 'state_dict' in checkpoint.keys():
        state_tmp.update(checkpoint['state_dict'])
    elif 'net' in checkpoint.keys():
        state_tmp.update(checkpoint['net'])
    else:
        state_tmp.update(checkpoint)
    state_tmp.update(b_ws)

    net.load_state_dict(state_tmp)
    return net

def load_model_cifar10(arch, bit_length, ck_path=''):
    model_path = config.model_root
    arch = arch + "_quan_mid"

    if 'ResNet' in arch:
        model = torch.nn.DataParallel(quan_resnet_cifar.__dict__[arch](10, bit_length))
    elif 'vgg' in arch:
        model = torch.nn.DataParallel(quan_vgg_cifar.__dict__[arch](bit_length))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if ck_path == '':
        ck_path = os.path.join(model_path, "model.th")
    ck_dict = torch.load(ck_path)
    model = load_state_dict(model, ck_dict)
    model.cuda()

    for m in model.modules():
        if isinstance(m, quan_Linear):
            m.__reset_stepsize__()
            m.__reset_weight__()
            weight = m.weight.data.detach().cpu().numpy()
            bias = m.bias.data.detach().cpu().numpy()
            # step_size = np.array([m.step_size.detach().cpu().numpy()])[0]
            step_size = np.float32(m.step_size.detach().cpu().numpy())
    return weight, bias, step_size

def load_model_imagenet(arch, bit_length, ck_path=''):
    arch = arch + "_quan_mid"
    if 'ResNet' in arch:
        model = torch.nn.DataParallel(quan_resnet_imagenet.__dict__[arch](num_classes=1000, n_bits=bit_length))
    elif 'vgg' in arch:
        model = torch.nn.DataParallel(quan_vgg_imagenet.__dict__[arch](bit_length))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        ck_dict = torch.load(ck_path)
    model = load_state_dict(model, ck_dict)
    model.cuda()

    for m in model.modules():
        if isinstance(m, quan_Linear):
            m.__reset_stepsize__()
            m.__reset_weight__()
            weight = m.weight.data.detach().cpu().numpy()
            bias = m.bias.data.detach().cpu().numpy()
            # step_size = np.array([m.step_size.detach().cpu().numpy()])[0]
            step_size = np.float32(m.step_size.detach().cpu().numpy())
    return weight, bias, step_size



def load_model(dc, arch, bit_length, ck_path):
    if dc == 'cifar10':
        return load_model_cifar10(arch, bit_length, ck_path)
    elif dc == 'imagenet':
        return load_model_imagenet(arch, bit_length, ck_path)

def load_data_cifar10(arch, bit_length, ck_path):
    mid_dim = {
        'ResNet18':1 * 512,
        'vgg16':512,
        'vgg19':512,
        'ResNet34':1 * 512,
        'ResNet50':4 * 512
    }[arch]
    arch = arch + "_quan_mid"
    if 'ResNet' in arch:
        model = torch.nn.DataParallel(quan_resnet_cifar.__dict__[arch](10, bit_length))
    elif 'vgg' in arch:
        model = torch.nn.DataParallel(quan_vgg_cifar.__dict__[arch](bit_length))
  
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    ck_dict = torch.load(ck_path)
    model = load_state_dict(model, ck_dict)
    model.cuda()

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    val_set = datasets.CIFAR10(root=config.cifar_root, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]), download=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=256, shuffle=False, pin_memory=True)

    mid_out = np.zeros([10000, mid_dim])
    labels = np.zeros([10000])
    start = 0
    model.eval()
    for i, (input, target) in tqdm(enumerate(val_loader)):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()

        # compute output before FC layer.
        output = model(input_var)
        mid_out[start: start + 256] = output.detach().cpu().numpy()

        labels[start: start + 256] = target.numpy()
        start += 256

    mid_out = torch.tensor(mid_out).float().cuda()
    labels = torch.tensor(labels).float()

    return mid_out, labels


def load_data(dc, arch, bit_length, ck_path):
    if dc == 'cifar10':
        return load_data_cifar10(arch, bit_length, ck_path)

def load_clean_output(bit_length, weight, bias, step_size, all_data, args):
    tmp_path = os.path.join(config.intermediate_results, f"{args.ck_path.split('/')[-1]}_{bit_length}_clean_output.pth")
    if not os.path.exists(tmp_path):
        auglag_st = augLag_Linear(bit_length, weight, bias, step_size, init=True).cuda()
        clean_output = auglag_st(all_data).detach().cpu().numpy()
        torch.save(clean_output, tmp_path)
    else:
        clean_output = torch.load(tmp_path)
    return clean_output

def load_auglag_st(bit_length, weight, bias, step_size, args):
    tmp_path = os.path.join(config.intermediate_results, f"{args.ck_path.split('/')[-1]}_{bit_length}_auglag_st.pth")
    if not os.path.exists(tmp_path):
        auglag_st = augLag_Linear(bit_length, weight, bias, step_size, init=True).cuda()
        torch.save(auglag_st, tmp_path)
    else:
        auglag_st = torch.load(tmp_path)
    return auglag_st
