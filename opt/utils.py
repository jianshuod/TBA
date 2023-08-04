import os 
import copy
import numpy as np

import torch

import config

def test_asr(auglag, all_data, target_idx, target_class):
    output_t = auglag(all_data[target_idx:target_idx+1])
    _, pred_t = output_t.topk(1, 1, True, True)
    pred_t = pred_t.squeeze(1)
    return target_class == pred_t[0].item()

def cal_acc(auglag, all_data, labels, target_idx, aux_idx, n_aux):
    output_a = auglag(all_data)
    _, pred_a = output_a.topk(1, 1, True, True)
    pred_a = pred_a.squeeze(1)
    pa_acc_a = len([i for i in range(len(output_a)) if labels[i] == pred_a[i] and i != target_idx and i not in aux_idx]) / \
                     (len(labels) - 1 - n_aux)
    return pa_acc_a

def cal_acc_asr(auglag, all_data, labels, target_idx, aux_idx, n_aux, target_class):
    output_a = auglag(all_data)
    _, pred_a = output_a.topk(1, 1, True, True)
    pred_a = pred_a.squeeze(1)
    pa_acc_a = len([i for i in range(len(output_a)) if labels[i] == pred_a[i] and i != target_idx and i not in aux_idx]) / \
                     (len(labels) - 1 - n_aux)
    return pa_acc_a, target_class == pred_a[target_idx].item()

def assemble_b(auglag, s_c, t_c):
    b_b_new_s = auglag.w_twos.data[s_c].view(-1).detach().cpu().numpy()
    b_b_new_t = auglag.w_twos.data[t_c].view(-1).detach().cpu().numpy()
    b_b = np.append(b_b_new_s, b_b_new_t)
    return b_b

def project_box(x):
    xp = x
    xp[x>1]=1
    xp[x<0]=0

    return xp


def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones(x.size)
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec

    return xp


def project_positive(x):
    xp = np.clip(x, 0, None)
    return xp

def get_binary_v(auglag, previous_auglag):
    '''

    :param auglag:
    :param previous_auglag:
    :return:
        flag denote whether the new version is the same with previous one
        True is different
        False is the same
    '''
    auglag_p = copy.deepcopy(auglag)
    auglag_p.w_twos.data[auglag.w_twos.data > 0.5] = 1.0
    auglag_p.w_twos.data[auglag.w_twos.data < 0.5] = 0.0
    n_bit = torch.norm(auglag_p.w_twos.data.view(-1) - previous_auglag.w_twos.data.view(-1),p=0).item()
    dif_flag = False if n_bit < 1 else True
    return auglag_p, dif_flag


def load_aux_idx(random_choice, attack_idx, total_len, args):
    if random_choice:
        return np.random.choice(
                [i for i in range(total_len) if i != attack_idx], args.n_aux,
                replace=False)
    else:
        # fix all baselines using the same auxiliary set
        return np.loadtxt(os.path.join(
            config.intermediate_results, f"aux_{args.dc}_{args.arch}_{args.bit_length}_{args.tn}_{args.n_aux}.txt"
        ), dtype=int)
