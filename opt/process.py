import os
import copy

import numpy as np

import torch

from .lf import loss_func_b_hat, loss_func_b_rel
from .utils import assemble_b, project_box, project_shifted_Lp_ball
from .utils import get_binary_v, test_asr, cal_acc


def attack_v2(auglag_st, attack_info, all_data, labels_cuda, aux_idx, clean_output,
            lambdas, args, log_handle=None):

    source_class, target_class, attack_idx = attack_info
    all_idx = np.append(aux_idx, attack_idx)
    print(lambdas, attack_idx, target_class, file=log_handle)

    # set hyperparams
    inn_lr_b_hat = args.inn_lr
    inn_lr_b_rel = args.inn_lr
    projection_lp = args.projection_lp
    rho_fact = args.rho_fact
    n_aux = args.n_aux
    margin = args.margin

    rho1 = args.initial_rho1
    rho2 = args.initial_rho2
    rho3 = args.initial_rho3
    rho4 = args.initial_rho4
    max_rho1 = args.max_rho1
    max_rho2 = args.max_rho2
    max_rho3 = args.max_rho3
    max_rho4 = args.max_rho4
    lam1, lam2, lam3, lam4, lam5 = lambdas
 
    # do copy
    auglag_a = copy.deepcopy(auglag_st)
    auglag_b = copy.deepcopy(auglag_st)
    auglag_a_t = copy.deepcopy(auglag_st)
    auglag_b_t = copy.deepcopy(auglag_st)

    # save some original states
    w_ori = auglag_st.w_twos.data.view(-1).clone().detach()

    input_var = torch.autograd.Variable(all_data[all_idx], volatile=True)
    target_var = torch.autograd.Variable(labels_cuda[all_idx].long(), volatile=True)
    target_cf_b_hat = clean_output[attack_idx][[i for i in range(len(clean_output[-1])) if i != source_class]].max() + margin
    target_cf_b_rel = clean_output[attack_idx][source_class]

    # prepare intermediate variables and residual variables
    b_a = assemble_b(auglag_a, source_class, target_class)
    y1 = b_a
    y2 = y1

    b_b = assemble_b(auglag_b, source_class, target_class)
    y3 = b_b
    y4 = y3

    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y2)
    z3 = np.zeros_like(y3)
    z4 = np.zeros_like(y4)

    counter_ab = 0
    asr_a, asr_b = False, False
    min_n_flip_ab = np.iinfo(np.int16).max
    asr_calculater = lambda x:test_asr(x, all_data, attack_idx, target_class)
    acc_calculater = lambda x:cal_acc(x, all_data, labels_cuda, attack_idx, aux_idx, n_aux)

    for ext_iter in range(args.ext_max_iters):

        # update b_hat
        for inn_iter in range(args.inn_max_iters):
            output = auglag_b(input_var)
            loss_b_hat, loss_items_b_hat = \
                loss_func_b_hat(output, target_var, target_cf_b_hat, source_class, target_class, b_a, auglag_b.w_twos, lam3, lam4, lam5, y3, y4, z3, z4, rho3, rho4)
            
            loss_b_hat.backward(retain_graph=True)
            auglag_b.w_twos.data[target_class] = auglag_b.w_twos.data[target_class] - inn_lr_b_hat * auglag_b.w_twos.grad.data[target_class]
            auglag_b.w_twos.data[source_class] = auglag_b.w_twos.data[source_class] - inn_lr_b_hat * auglag_b.w_twos.grad.data[source_class]
            auglag_b.w_twos.grad.zero_()
        
        b_b = assemble_b(auglag_b, source_class, target_class)

        # update intermediate variables y3, y4
        y3 = project_box(b_b + z3 / rho3)
        y4 = project_shifted_Lp_ball(b_b + z4 / rho4, projection_lp)

        temp3 = np.linalg.norm(b_b - y3) / max(np.linalg.norm(b_b), 2.2204e-16)
        temp4 = np.linalg.norm(b_b - y4) / max(np.linalg.norm(b_b), 2.2204e-16)

        # do log
        if ext_iter % args.log_interval == 0:
            # print('[B_hat] iter: %d, counter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, counter_ab, max(temp3, temp4), loss_b_hat.item()), loss_items_b_hat)
            auglag_a_t, dif_flag_a = get_binary_v(auglag_a, auglag_a_t)
            auglag_b_t, dif_flag_b = get_binary_v(auglag_b, auglag_b_t)
            if dif_flag_a: asr_a = asr_calculater(auglag_a_t)
            if dif_flag_b: asr_b = asr_calculater(auglag_b_t)
            if (dif_flag_a or dif_flag_b or 0 == ext_iter) and (asr_a == False and asr_b == True):
                n_bit_b = torch.norm(auglag_a_t.w_twos.data.view(-1) - auglag_b_t.w_twos.data.view(-1), p=0).item()
                counter_ab = 0
                if n_bit_b < min_n_flip_ab:
                    min_n_flip_ab = n_bit_b
                    temp_a = copy.deepcopy(auglag_a_t)
                    temp_b = copy.deepcopy(auglag_b_t)
            else: counter_ab += args.log_interval # roughly estimate

        # update b_rel
        for inn_iter in range(args.inn_max_iters):
            output = auglag_a(input_var)
            loss_b_rel, loss_items_b_rel = \
                loss_func_b_rel(output, target_var, target_cf_b_rel, source_class, target_class, b_b, auglag_a.w_twos, lam3, lam1, lam2, y1, y2, z1, z2, rho1, rho2)
            loss_b_rel.backward(retain_graph=True)
            auglag_a.w_twos.data[target_class] = auglag_a.w_twos.data[target_class] - inn_lr_b_rel * auglag_a.w_twos.grad.data[target_class]
            auglag_a.w_twos.data[source_class] = auglag_a.w_twos.data[source_class] - inn_lr_b_rel * auglag_a.w_twos.grad.data[source_class]
            auglag_a.w_twos.grad.zero_()
        b_a = assemble_b(auglag_a, source_class, target_class)

        # update y1, y2
        y1 = project_box(b_a + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_a + z2 / rho2, projection_lp)

        temp1 = np.linalg.norm(b_a - y1) / max(np.linalg.norm(b_a), 2.2204e-16)
        temp2 = np.linalg.norm(b_a - y2) / max(np.linalg.norm(b_a), 2.2204e-16)

        # do log
        # if ext_iter % args.log_interval == 0:
        #     print('[B_Rel] iter: %d, counter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, counter_ab, max(temp1, temp2), loss_b_rel.item()), loss_items_b_rel)

        # update residual variables
        z1 = z1 + rho1 * (b_a - y1)
        z2 = z2 + rho2 * (b_a - y2)
        z3 = z3 + rho3 * (b_b - y3)
        z4 = z4 + rho4 * (b_b - y4)

        # update rho
        if ext_iter % args.rho_refresh_int == 0:
            rho1 = min(rho_fact * rho1, max_rho1)
            rho2 = min(rho_fact * rho2, max_rho2)
            rho3 = min(rho_fact * rho3, max_rho3)
            rho4 = min(rho_fact * rho4, max_rho4)        

        # judge if early stop
        # case 1: failure to explorem, NaN occurs
        if True in np.isnan(b_a) or True in np.isnan(b_b):
            break 
        # case 2: lp-box ADMM converges(y1->b_a, y2->b_a or y3->b_b, y4->b_b)
        if max(temp1, temp2) <= args.stop_threshold and max(temp3, temp4) <= args.stop_threshold and ext_iter > 100:
            print('[Break Point1] END iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp3, temp4), loss_b_hat.item()))
            print('[Break Point2] END iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp1, temp2), loss_b_rel.item()))
            break
        # case 3: no improvement for b_b
        if ext_iter >= args.ext_min_iters and counter_ab >= args.counter_tolerance:
            print("No improvement! Early Stop")
            break
    if min_n_flip_ab == np.iinfo(np.int16).max:
        res = (-1, 0, 100, 0, 100)
        M_a = None
    else:
        n_bit_a = torch.norm(temp_a.w_twos.data.view(-1) - w_ori, p=0).item()
        acc_a = acc_calculater(temp_a)
        acc_b = acc_calculater(temp_b)
        res = (1, acc_a, n_bit_a, acc_b, min_n_flip_ab)

        M_a = temp_a
    print(res)
    return res, ext_iter, M_a