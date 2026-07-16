import torch
class SoftBinning:
    def __init__(self, bins=3):
        if bins < 3 or bins % 2 != 1:
            raise ValueError("bins must be odd >= 3")
        self.bins = bins // 2
        self.max = self.bins

    @torch.no_grad()
    @torch.compile()
    def call_test(self, x:torch.Tensor,factor:torch.Tensor):
        x = torch.abs(x) / factor
        y = 1 - self.max + x
        mask = x < self.max
        y[mask] = x[mask] - torch.floor(x[mask])
        y = y * torch.abs(1 -y)
        return y
    def compute_xright_quantile(self,x:torch.Tensor):
        x ,_indices= x.clone().detach().flatten().sort()
        third = x[x.shape[0] // 3]
        two_thirds = x[int(x.shape[0] * (2/3))]
        return (third.abs() + two_thirds.abs() ) #  /2 *2 (average of abs of third and two thirds, times two 
        # to get x_right)
# soft_binning = SoftBinning(bins=3)

import time
@torch.compile()
@torch.no_grad()
def calc_reg_loss(optimizer,soft_binning,n_params,device= torch.device("cuda:0"),M = 1,beta = 5e-4):
    """
    factor_metric is mean of mean of tensor.
    returns tuple of:
    avg_reg_loss, a norm, average_factor, %done
    """
    factor_iter = iter(optimizer.state["factors"])
    reg_loss = torch.zeros((),device=device,requires_grad=False)
    quantized_params = torch.zeros((),device=device,requires_grad=False,dtype=torch.int64)
    reg_num_parms = torch.zeros((),device=device,requires_grad = False)
    norm = torch.zeros((),device=device,requires_grad=False)
    factor_metric = torch.zeros((),device=device,requires_grad=False)
    for group in optimizer.get_regularized_param_groups():
        if group["quant_bits"] == 0:
            for param in group["params"]:
                next_factor = next(factor_iter)
                reg_loss += soft_binning.call_test(param.reshape(param.shape[0],-1), next_factor).sum().detach()
                
                scaled_param = param.data.flatten(1) / next_factor
                quantized_params+=  (abs(scaled_param - scaled_param.round().clip(-M,M)) < (beta/2)).sum().to(torch.int64)
                reg_num_parms+=1.
                norm += param.norm().detach()
                factor_metric += next_factor.mean()
    return (reg_loss / n_params) .item(), norm.sqrt().item(), factor_metric.item() / len(optimizer.state["factors"]),quantized_params / n_params


# def calc_numel(optimizer):
#     numel = 0
#     for group in optimizer.get_regularized_param_groups():
#         if group["quant_bits"] == 0:
#             for param in group["params"]:
                
                