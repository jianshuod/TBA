import warnings
warnings.filterwarnings("ignore")
import config

import os
print(f"pid {os.getpid()}")
os.chdir(config.default_path)

import time
import argparse
import numpy as np

from models.quantization import *
from opt.process import  attack_v2
from utils.utils import load_model, load_data, initialize_env, load_auglag_st, load_clean_output
from opt.utils import load_aux_idx

parser = argparse.ArgumentParser(description='TBA')
parser.add_argument('-bw', default=8, type=int, help='bit width')
parser.add_argument('-mc', default='ResNet18', type=str, help='Model choice')
parser.add_argument('-dc', default='cifar10', type=str, help='Dataset choice')
parser.add_argument('-tn', default=1000, type=int, help='number of target instances')
parser.add_argument("--rc",action="store_true",help="randomly select aux set")

parser.add_argument("--manual",action="store_true",help="Mannually set hyperparams")
parser.add_argument("-bs_l", type=float, default=1, help="base lambda")
parser.add_argument("-ri", type=float, default=1, help="ratio inner")
parser.add_argument("-ro", type=float, default=15, help="ratio outer")
parser.add_argument("-lr", type=float, default=0.005, help="inner learning rate")
opt = parser.parse_args()

def main():
    initialize_env(config.seed)

    # prepare the data
    print("Prepare data ... ")

    args = config.args(opt)
    args.show_args()

    ck_path = args.ck_path
    arch = args.arch
    bit_length = args.bit_length

    weight, bias, step_size = load_model(args.dc, arch, bit_length, ck_path=ck_path)

    all_data, labels = load_data(args.dc, arch, bit_length, ck_path)
    labels_cuda = labels.cuda()

    clean_output = load_clean_output(bit_length, weight, bias, step_size, all_data, args)

    target_attk_insts_path = args.target_atk_insts

    attack_info = np.loadtxt(target_attk_insts_path).astype(int)
    total = 0
    suc_num = 0
    total_n_bit_ab = 0
    total_acc_a = 0
    total_acc_b = 0
    total_time = 0
    total_n_bit_a_ori = 0
    n_bit_abs = []
    acc_as = []
    acc_bs = []
    n_bit_a_oris = []
    time_costs = []
    suc_insts = []
    suc_targs = []
    end_iters = []
    total_len = len(attack_info)
    for i, (target_class, attack_idx) in enumerate(attack_info):
        print('--------------------------------------------')
        source_class = int(labels[attack_idx])
        total += 1
        aux_idx = load_aux_idx(args.rc, attack_idx, len(labels), args)

        s_time = time.time()

        print("Prepare a start point")
        auglag_st = load_auglag_st(bit_length, weight, bias, step_size, args)
        l1, l2, l3, l4, l5 = (args.lam1, args.lam2, args.lam3, args.lam4, args.lam5)

        print("Attack_alpha Start")
        (select_iter, max_acc_a, select_n_bit_a_ori, max_acc_b, min_n_bit_ab), end_iter,_ \
            = attack_v2(auglag_st, (source_class, target_class, attack_idx), all_data, labels_cuda, aux_idx, clean_output, (l1, l2, l3, l4, l5), args)

        this_time = time.time() - s_time
        total_time += this_time
        print(f"{target_class}, {attack_idx}, time_cost {this_time}")
        if select_iter != -1:
            suc_num += 1
            total_n_bit_ab += min_n_bit_ab
            total_n_bit_a_ori += select_n_bit_a_ori
            total_acc_a += max_acc_a
            total_acc_b += max_acc_b
            suc_insts.append(attack_idx)
            suc_targs.append(target_class)
            n_bit_abs.append(min_n_bit_ab)
            n_bit_a_oris.append(select_n_bit_a_ori)
            acc_as.append(max_acc_a * 100)
            acc_bs.append(max_acc_b * 100)
            time_costs.append(this_time)
            end_iters.append(end_iter)
        if suc_num == 0: continue
        else:
            print(f"Total {total}, suc {suc_num}, fail {total - suc_num}, avg_n_bit_a_ori {total_n_bit_a_ori / suc_num} avg_n_bit_if_suc {total_n_bit_ab / suc_num}, "
            f"avg_acc_a {total_acc_a / suc_num}, avg_acc_b {total_acc_b / suc_num}, avg_time_cost {total_time / total}")

    print(f'ASR {suc_num / total_len * 100}')
    print(f'N_bit_a_ori mean {np.average(n_bit_a_oris)} std {np.std(n_bit_a_oris)}')
    print(f'ACC_a mean {np.average(acc_as)} std {np.std(acc_as)}')
    print(f'N_bit_a_b mean {np.average(n_bit_abs)} std {np.std(n_bit_abs)}')
    print(f'ACC_b mean {np.average(acc_bs)} std {np.std(acc_bs)}')
    print(f'Exterior iterations mean {np.average(end_iters)}')
    print(f'Time Cost {np.average(time_costs)}')

if __name__ == "__main__":
    main()
