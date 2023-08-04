import torch
import torch.nn.functional as F

def loss_func_b_hat(output, labels, target_h_thr, s, t, b_rel, w, lam3, lam4, lam5, y3, y4, z3, z4, rho3, rho4):
    l1 = F.cross_entropy(output[:-1], labels[:-1]) * lam4

    l2_1 = torch.max(output[-1][s] - output[-1][t], torch.tensor(0.0).cuda())
    l2_2 = torch.max(target_h_thr - output[-1][t], torch.tensor(0.0).cuda())
    l2 = (l2_1 + l2_2) * lam5

    y3, y4, z3, z4= torch.tensor(y3).float().cuda(), torch.tensor(y4).float().cuda(),\
                    torch.tensor(z3).float().cuda(), torch.tensor(z4).float().cuda()
    
    b_rel = torch.tensor(b_rel).float().cuda()
    b_hat = torch.cat((w[s].view(-1), w[t].view(-1)))
    l3 = torch.norm(b_hat - b_rel) ** 2 * lam3

    l4 = (rho3/2) * torch.norm(b_hat - y3) ** 2 + (rho4/2) * torch.norm(b_hat - y4) ** 2\
        + z3@(b_hat-y3) + z4@(b_hat-y4)
    
    return l1 + l2 + l3 + l4, [l1.item(), l2.item(), l3.item(), l4.item()]

def loss_func_b_rel(output, labels, source_h_thr, s, t, b_hat, w, lam3, lam1, lam2, y1, y2, z1, z2, rho1, rho2):

    l1 = F.cross_entropy(output[:-1], labels[:-1]) * lam1

    l2_1 = torch.max(source_h_thr - output[-1][s], torch.tensor(0.0).cuda())
    l2_2 = torch.max(output[-1][t] - output[-1][s], torch.tensor(0.0).cuda())
    l2 = (l2_1 + l2_2) * lam2

    y1, y2, z1, z2= torch.tensor(y1).float().cuda(), torch.tensor(y2).float().cuda(),\
                    torch.tensor(z1).float().cuda(), torch.tensor(z2).float().cuda()

    b_hat = torch.tensor(b_hat).float().cuda()
    b_rel = torch.cat((w[s].view(-1), w[t].view(-1)))
    l3 = torch.norm(b_hat - b_rel)**2 * lam3

    l4 = (rho1/2) * torch.norm(b_rel - y1) ** 2 + (rho2/2) * torch.norm(b_rel - y2) ** 2\
        + z1@(b_rel-y1) + z2@(b_rel-y2)
  
    return l1 + l2 + l3 + l4, [l1.item(), l2.item(), l3.item(), l4.item()]