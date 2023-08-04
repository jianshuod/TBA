import os

cifar_root = 'xxx'
imagenet_root = 'xxx'

default_path = 'xxx'
model_root = os.path.join(default_path, 'checkpoints/model_ck')
intermediate_results = os.path.join(default_path, 'intermediate_results')
log_root = os.path.join(default_path, 'logs/tba')
target_insts_path = os.path.join(default_path, 'atk_insts')
attk_insts_path = os.path.join(default_path, 'atk_insts')
seed = 512

no_use_gpu_id = []

class args():
    gpu_id = '1'
    attack_idx = 1317
    target_class = 0
    log_interval = 1

    base_lambda = 1
    ratio_in = 10
    ratio_out = 1.5

    margin = 10

    ext_max_iters = 2000
    ext_min_iters = 1000
    inn_max_iters = 3
    initial_rho1 = 0.0001
    initial_rho2 = 0.0001
    initial_rho3 = 0.0001
    initial_rho4 = 0.0001
    max_rho1 = 50
    max_rho2 = 50
    max_rho3 = 50
    max_rho4 = 50
    rho_fact = 1.01
    rho_refresh_int = 1
    stop_threshold = 1e-4
    counter_tolerance = 300
    projection_lp = 2

    def __init__(self, opt) -> None:
        self.dc = opt.dc
        self.rc = opt.rc
        self.arch = opt.mc
        self.bit_length = opt.bw
        self.ck_path = os.path.join(default_path, 
           {
            'cifar10': 
                {
                    'ResNet18': 'checkpoint/resnet18/176_95.25.pth',
                    'vgg16':'checkpoint/vgg16/182_93.64.pth'
                },
            'imagenet':
                {
                    'ResNet34': 'checkpoint/model_pth_imagenet/resnet34-b627a593.pth',
                    'vgg19_bn':'checkpoint/model_pth_imagenet/vgg19_bn-c79401a0.pth'
                }
            }[self.dc][self.arch]
        )
        if hasattr(opt, 'manual') and opt.manual:
            self.base_lambda = opt.bs_l
            self.ratio_in = opt.ri
            self.ratio_out = opt.ro
            self.inn_lr = opt.lr

        self.lam1 = self.base_lambda
        self.lam2 = self.ratio_in * self.lam1
        self.lam3 = self.ratio_out * self.lam1
        self.lam4 = self.base_lambda
        self.lam5 = self.ratio_in * self.lam4
        self.tn = opt.tn
        self.target_atk_insts = os.path.join(attk_insts_path, f'{self.dc}_atks.txt')

        self.n_aux = 128 if self.dc == 'cifar10' else 512
        self.inn_lr = 0.005 if self.dc == 'cifar10' else 0.01


    def show_args(self):
        property = [f"{p}:{getattr(self, p)}" for p in dir(self) if'__' not in p]
        print(', '.join(property))


