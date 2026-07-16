# adapted from parq paper code, copyright is therefor by meta, apple, under MIT
# and by me (david edel)
# papers: TODO: link parq, lsbq
import torch
def binary_sign(input: torch.Tensor) -> torch.Tensor: 
    """
    from paper for lsbq
    Same as `torch.sign(input)` but map 0 to 1.
    
    """
    return torch.where(input == 0, 1.0, input.sign())

from typing import Iterable, Self
from torch.optim import Optimizer
import itertools
import torch
import logging

def tanh_list(params,factors,beta):
    for param, factor in zip(params,factors):
        param.copy_()

def piece_quad_prox(params,factors,beta):
    if beta < 1.0:
        apply_prox_to_list_optim(
            params,
            factors,
            beta,
        )
    else:
        apply_prox_hard_rounding_list(
            params,
            factors,
            beta=beta
        )

@torch.compile()
@torch.no_grad()
def apply_prox_to_list_general(params, factors, beta):
    """
    this uses a slow implementation, avoid if if you know if beta is larger or smaller than 1.
    """
    for param, factor in zip(params, factors):
        param.copy_(
            apply_prox_same_beta_general(
                param.data, beta, torch.tensor(1.0), factor
            ).detach()
        )

@torch.compile()
@torch.no_grad()
def apply_prox_hard_rounding_list(params,factors,beta):
    for param, factor in zip(params,factors):
        param.copy_(apply_hard_rounding_ternary(param.data,factor).detach())


@torch.compile()
@torch.no_grad()
def apply_prox_to_list_optim(params, factors, beta):
    """
    assume beta < 1 and M = 1
    """
    for param, factor in zip(params, factors):
        param.copy_(
            apply_prox_same_beta_optim(param.data, beta, factor).detach()
        )


@torch.compile(dynamic=False)
@torch.no_grad()
def apply_prox_same_beta_optim(
    theta_old: torch.Tensor, beta: torch.Tensor, factors: torch.Tensor
):
    # factor replaces the one
    # this means that factor * one = factor, which is where the slope will be 0 after the transformation
    """
    assume beta < 1 and M = 1
    will handle every tensor row-wise, so (A,B,C) will be treated as (A,B*C)
    """
    # print(theta_old.shape,factors.shape)
    theta_shape = theta_old.shape
    theta_old = theta_old.flatten(1)
    a0 = (beta / 2) * factors
    abs_theta = abs(theta_old)
    y = torch.where(abs_theta < a0, 0, abs_theta / (1 - beta) - (1 / (1 - beta) * a0))
    # y[abs_theta>(1-a0)]=1
    y = torch.where(abs_theta > (factors - a0), factors, y)
    y = torch.where(
        abs_theta > factors + a0,
        abs_theta / (1 + beta) + (factors - ((1 + beta / 2) * factors / (1 + beta))),
        y,
    )  # factors - ((1+ beta/2) * factors / (1+beta))
    return (y * torch.sign(theta_old)).reshape(theta_shape)


@torch.compile()
@torch.no_grad()
def apply_hard_rounding_ternary(theta_old:torch.Tensor,factor:torch.Tensor):
    theta_shape = theta_old.shape
    theta_flat = theta_old.flatten(1)
    theta_flat = theta_flat / factor
    theta_flat = theta_flat.clip(-1.,1.).round()
    return (theta_flat * factor).reshape(theta_shape)


@torch.compile()
@torch.no_grad()
def prox_piece_quad_small_beta(
    theta_old: torch.Tensor, beta: torch.Tensor, factors: torch.Tensor
):  
    """
    assume beta < 1
    M can be any integer >= 1
    For M = 1, there is specialized version, why depending on optimization could be faster
    will handle every tensor row-wise, so (A,B,C) will be treated as (A,B*C)
    """

    theta_shape = theta_old.shape # will be used as output shape at the end
    theta_old = theta_old.flatten(1) 
    # because we have a rowwise scaled matrix
    # if matrix has higher dim, assume we still have scaling over dim 0 (correct for conv weights)

    theta_trunc = theta_old.trunc() # round towards 0. This enables us to calculate as if all values are between -1 and 1
    # this will then later be added again

    theta_old = theta_old - theta_trunc 
    #theta_old -= theta_trunc
    a0 = (beta / 2) * factors # calculate the x at which the ascending line begins
    abs_theta = abs(theta_old) # handle negative vales as if they are positive. later add the sign again
    y = torch.where(abs_theta < a0, 0, abs_theta / (1 - beta) - (1 / (1 - beta) * a0))
    y = torch.where(abs_theta > (factors - a0), factors, y)
    y = torch.where(
        abs_theta > factors + a0,
        abs_theta / (1 + beta) + (factors - ((1 + beta / 2) * factors / (1 + beta))),
        y,
    )  # factors - ((1+ beta/2) * factors / (1+beta))
    return ((y * torch.sign(theta_old)) + theta_trunc) .reshape(theta_shape)



@torch.compile()
@torch.no_grad()
def apply_prox_same_beta_general(
    theta_old: torch.Tensor, beta: torch.Tensor, M: torch.Tensor, factor
):
    """
    beta is the same for all params
    can handle any M (M=1 for ternary)
    will handle every tensor row-wise, so (A,B,C) will be treated as (A,B*C)
    can handle any beta
    """
    theta_shape = theta_old.shape
    theta_old = theta_old.flatten(1)
    ONE = torch.tensor(1.0)
    s = torch.sign(theta_old / factor)
    u = abs(theta_old) / factor
    mask_in = u < M

    u_m = torch.floor(u)
    t_in = u - u_m
    theta_new = torch.zeros_like(theta_old)
    if beta >= ONE:
        # if abs(s) < M: apply this:
        theta_new_in = s * torch.round(u)
        theta_new[mask_in] = theta_new_in[mask_in]
    else:  # beta < 1
        theta_new_in = s * (
            torch.clip((t_in - beta / 2) / (1 - beta), 0, 1) + u_m
        )  # t_in - beta in brackets??
        theta_new[mask_in] = theta_new_in[mask_in]
    # for all     abs(param) > M:
    t_out = u - M
    theta_new_out = s * (torch.nn.functional.relu((t_out - beta / 2) / (1 + beta)) + M)
    theta_new[~mask_in] = theta_new_out[~mask_in]
    return (theta_new * factor).reshape(theta_shape)

#@torch.compile()
@torch.no_grad()
def lsbq_ternary(x:torch.Tensor) -> torch.Tensor:
    x = x.flatten(1)
    v_cands = x.abs().sort(dim=1).values # dim =1 is equal to dim = -1 , flatten conv to (x,-1)
    cumsum = v_cands.cumsum(dim=1)
    cumsum, total_sum = cumsum[:,1:-1],cumsum[:,-1:]
    counts = torch.arange(1,x.size(dim=1),device=x.device)
    counts_r2l = counts[:-1].flip((-1,))
    cmean_r2l = (total_sum - cumsum).div_(counts_r2l.mul_(2))
    v_cands, v_cands2 = v_cands[:,1:-1], v_cands[:,2:]

    mask = (v_cands <= cmean_r2l).logical_and_(v_cands2 >= cmean_r2l)
    optimal_v = x.mean(dim=1,keepdim=True).div_(2)
    row_invalid = optimal_v < x.min(dim=1,keepdim=True).values
    if row_invalid.any():
        extra_col = row_invalid.to(x.dtype).mul(optimal_v)
        v_cands = torch.cat((v_cands, extra_col), -1)
        mask = torch.cat((mask, row_invalid), -1)
    split_sizes = mask.sum(dim=1).tolist()
    v_cands = v_cands[mask].split(split_sizes)
    v_cands = torch.nested.nested_tensor(list(v_cands))
    v_cands = torch.nested.to_padded_tensor(v_cands, 0)

    r = x.unsqueeze(-2)
    v = v_cands.unsqueeze(-1)
    r = r.sub(v*binary_sign(r))
    r = r.sub(v * binary_sign(r))
    costs = r.norm(dim=-1)
    indices = costs.argmin(dim=-1,keepdim=True)
    v_best = v_cands.gather(1,indices)
    v_best = v_best.flatten()
    v_best = torch.where(v_best==0.,x.abs().mean(1) * 0.7,v_best )
    return v_best * 2

# wtf is this
# optimal_v[row_invalid] = x[row_invalid].mean(dim=1,keepdim=True).div_(2)
# optimal_v[optimal_v.isnan()] = 0
# optimal_v = torch.where(mask.any(dim=1,keepdim=True),v_cands[mask].view(-1,1),optimal_v)
# optimal_v = torch.where(optimal_v.isnan(),0,optimal_v)


#TODO: test if this can be compiled
@torch.no_grad()
def equisplit(x,percentile = 1/3):
    """
    calcualtes factor based on quantiles
    sorts the matrix in each row
    takes the percenile * N th element and the (1- percentiles) * N th element and retuns their mean of abs
    

    x: at least 2 dim tensor
    percentile: 
    """
    x = x.flatten(1)
    x_sorted ,_= x.detach().sort(1) # now per row scaling 
    x_sorted: torch.Tensor
    third = x_sorted[:,int(x_sorted.shape[1] * percentile),None] # None for keepdim
    two_thirds = x_sorted[:,int(x_sorted.shape[1] * (1 - percentile)),None] 
    equi_spit = (third.abs() + two_thirds.abs() )
    return equi_spit



def prox_piece_quad_small_beta_M(
    theta_old: torch.Tensor, beta: torch.Tensor, factors: torch.Tensor,M = 1
):
    # factor replaces the one
    # this means that factor * one = factor, which is where the slope will be 0 after the transformation
    """
    assume beta < 1
    M can be any integer >= 1
    For M = 1, there is specialized version, why depending on optimization could be faster
    will handle every tensor row-wise, so (A,B,C) will be treated as (A,B*C)

    this is currently not used, and needs some adaptation to be used.
    remember that you also ned a new compute_factors function for higher M. (std based probably good)
    """
    M = M - 1
    theta_shape = theta_old.shape # will be used as output shape at the end

    theta_old = theta_old.flatten(1)
    theta_sign = theta_old.sign()
    # because we have a rowwise scaled matrix
    # if matrix has higher dim, assume we still have scaling over dim 0 (correct for conv weights)


    abs_theta = abs(theta_old) # handle negative vales as if they are positive. later add the sign again
    theta_trunc_masked = torch.where(abs_theta < M,abs_theta.trunc(),M) # round towards 0. This enables us to calculate as if all values are between -1 and 1
    # this will then later be added again

    abs_theta = abs_theta - theta_trunc_masked 
    #theta_old -= theta_trunc
    a0 = (beta / 2) * factors # calculate the x at which the ascending line begins
    y = torch.where(abs_theta < a0, 0, abs_theta / (1 - beta) - (1 / (1 - beta) * a0))
    y = torch.where(abs_theta > (factors - a0), factors, y)
    y = torch.where(
        abs_theta > factors + a0,
        abs_theta / (1 + beta) + (factors - ((1 + beta / 2) * factors / (1 + beta))),
        y,
    )  # factors - ((1+ beta/2) * factors / (1+beta))
    return ((y  + theta_trunc_masked.clamp(-M,M) )* theta_sign) .reshape(theta_shape)






class SSTQO(Optimizer):
    """
    Stochastic Selective Ternary Quantization Optimizer
    NOT WORKING YET,
    DEVELOPMENT PAUSED
    """
    steps: int

    def __init__(
        self,
        base_optimizer: Optimizer,
        steps_per_epoch,
        reg_wait_epochs: int = 5,
        regularization_epochs=10,
        beta_schedule_iter: Iterable = None,
        device=torch.device("cuda:0"),
        logger=logging.getLogger(__name__),
    ):
        self.logger = logger
        super().__init__(
            [{"params": []}], {"lr": base_optimizer.defaults["lr"]}
        )  # NOTE  check if params from base_optimizer should be placed here
        self.base_optimizer = base_optimizer
        self.beta = 0  # TODO
        self.M = torch.tensor([1])
        self.state["steps"] = torch.zeros((), requires_grad=False)
        self.warmup_epochs = reg_wait_epochs
        self.total_steps = (regularization_epochs + reg_wait_epochs) * steps_per_epoch
        self.total_reg_steps = regularization_epochs * steps_per_epoch
        self.step_start_reg = reg_wait_epochs * steps_per_epoch
        self.beta_schedule = (
            beta_schedule_iter
            if beta_schedule_iter is not None
            else itertools.cycle([8e-5])
        )
        self.state["factors"]= []
        self.row_levels_inverse: list[torch.tensor] = []  # list of vectors for scaling
        self.regularized_params: list[torch.Tensor] = []
        for group in self.get_regularized_param_groups():
            self.regularized_params.extend(group["params"]) # append all reg parms to list for efficient
            # iteration later in compiled functions
            for param in group["params"]:
                # lets hope the ordering of this stays the same
                self.state["factors"].append(torch.ones((param.shape[0],), device=device))
                # self.row_levels_inverse.append(
                #     torch.ones((param.shape[0],), device=device)
                # )  # this is a inverse_levels tensor,
                # matricies get multiplied x * W^T, so the first dim of W is the amount of rows.
                # so for each matrix (for now only 2d weights), this will give each row( the weights for one neuron) a level.
                # The Quantization targets for this row will be {-level, 0, level}
                # when quantizing for real, multiply the row by 1/ level, and the bias by 1/ level
                #  scaling neurons will be inserted after the linear layer with a value of level.
        self.param_groups = self.base_optimizer.param_groups
        n_reg_groups = len(list(self.get_regularized_param_groups()))
        if n_reg_groups > 1:
            logger.warning(
                "Multiple regularized groups detected, this is maybe not supported yet."
            )

    def is_quantization_enabled(self):
        if self.state["steps"] <= self.step_start_reg:
            return False
        return True

    def get_regularized_param_groups(self):
        for group in self.base_optimizer.param_groups:
            if group.get("quant_bits", 16) < 16:
                yield group

    @torch.no_grad()
    def compute_factors_lsbq(self):
        """
        ternary version
        uses lsbq, copyright Apple MIT
        """
        self.logger.info("COMPUTING FACTORS: LSQB")
        # self.row_levels_inverse.clear()
        
        factors_iter = iter(self.state["factors"])
        
        
        for group in self.get_regularized_param_groups():
            if group["quant_bits"] == 0:  # we call ternary 2 bit, they do 0 bit
                for x in group["params"]:
                    factor = lsbq_ternary(x.data.detach()).flatten()
                    is_any_nan = factor.isnan().any().item()
                    is_any_0 = (factor==0.).any().item()
                    if is_any_nan or is_any_0:
                        self.logger.info(f"{is_any_nan=} and {is_any_0=}")

                    next(factors_iter).copy_(factor)
                    # self.row_levels_inverse.append(
                    #     (third.abs() + two_thirds.abs()).detach()
                    # )
                    # print(x.shape)
        self.logger.info("END FACTORS COMPUTE")

    @torch.no_grad()
    def compute_factors_equisplit(self):
        """

        """
        self.logger.info("COMPUTING FACTORS: equisplit")
        factors_iter = iter(self.state["factors"])
        for group in self.get_regularized_param_groups():
            if group["quant_bits"] == 0:  # we call ternary 2 bit, they do 0 bit
                for x in group["params"]:
                    
                    
                    factor = equisplit(x.data.detach())
                    #self.state["factors"].append(factor)
                    next(factors_iter).copy_(factor) # copy_ instead of reassign, to assist compiler
        self.logger.info("END FACTORS COMPUTE")


    def set_final_beta(self):
        """
        sets all following betas to 1 and disables requires_grad on regParams
        """
        self.beta_schedule = itertools.repeat(1.0)
        for param in self.regularized_params:
            param.requires_grad = False # is will disable gradient descent, but this will most likely 
            # not change anything, because beta is now 1 and hard rounding gets used.

    def step(self: Self):
        """

        preprocess the .grad of parameters by selectively setting to 0
        Idea: TODO: Let our implementation handle weight decay instead of the base optimizer 
        because that does not know about to probability of movement
        
        TODO: if often 0, momentum will go crazy, not good.
        Lets try anyway
        """
        gen = self.beta_schedule
        self.beta = next(gen)
        if self.is_quantization_enabled():
            apply_direction_scaling_list(self.regularized_params,self.state["factors"])
        else:
            pass  # continues with prox operator
        self.base_optimizer.step()
        self.state["steps"] += 1

@torch.compile()
@torch.no_grad()
def apply_direction_scaling_list(params,factors):
    for param , factor in zip(params,factors):
        param:torch.nn.Parameter
        apply_direction_scaling(param,factor)
                


@torch.compile()
@torch.no_grad()
def apply_direction_scaling(param:torch.nn.Parameter,factor:torch.Tensor):
    param_shape = param.shape
    scaled_param = (param.data.flatten(1) / factor).reshape(param_shape) # will be a view (hopefully)
    scaled_target = scaled_param.clip(-1.,1.).round()
    distance_target : torch.Tensor =  scaled_target - scaled_param # dist to {-1,0,1}
    # if scaled param values are not bigger than 1.5 (< -1.5)
    # this can only <= 0.5
    # The idea is to apply gradients more often if they point in the direction of target
    
    # distance_target:
    #  positive : target is bigger than param => increase param is good
    grad_right_direction_mask = (distance_target * param.grad) >= 0
    grad_right_direction_mask.where(grad_right_direction_mask == 1, 0.01)
    # TODO: think about 0 sign for both
    param.grad.mul_(grad_right_direction_mask)
    return





class TQP(Optimizer):
    """
    TODO: make serializable, (savable, restoreable)

    """

    steps: int

    def __init__(
        self,
        base_optimizer: Optimizer,
        steps_per_epoch,
        reg_wait_epochs: int = 5,
        regularization_epochs=10,
        beta_schedule_iter: Iterable = None,
        device=torch.device("cuda:0"),
        logger=logging.getLogger(__name__),
        reg_function_ = piece_quad_prox,
    ):
        self.reg_function_ = reg_function_
        self.logger = logger
        super().__init__(
            [{"params": []}], {"lr": base_optimizer.defaults["lr"]}
        )  # NOTE  check if params from base_optimizer should be placed here
        self.base_optimizer = base_optimizer
        self.beta = 0  # TODO
        self.M = torch.tensor([1])
        self.state["steps"] = torch.zeros((), requires_grad=False)
        self.warmup_epochs = reg_wait_epochs
        self.total_steps = (regularization_epochs + reg_wait_epochs) * steps_per_epoch
        self.total_reg_steps = regularization_epochs * steps_per_epoch
        self.step_start_reg = reg_wait_epochs * steps_per_epoch
        self.beta_schedule = (
            beta_schedule_iter
            if beta_schedule_iter is not None
            else itertools.cycle([8e-5])
        )
        self.state["factors"]= []
        self.row_levels_inverse: list[torch.tensor] = []  # list of vectors for scaling
        self.regularized_params: list[torch.Tensor] = []
        for group in self.get_regularized_param_groups():
            self.regularized_params.extend(group["params"]) # append all reg parms to list for efficient
            # iteration later in compiled functions
            for param in group["params"]:
                # lets hope the ordering of this stays the same
                self.state["factors"].append(torch.ones((param.shape[0],1), device=device))
                # self.row_levels_inverse.append(
                #     torch.ones((param.shape[0],), device=device)
                # )  # this is a inverse_levels tensor,
                # matricies get multiplied x * W^T, so the first dim of W is the amount of rows.
                # so for each matrix (for now only 2d weights), this will give each row( the weights for one neuron) a level.
                # The Quantization targets for this row will be {-level, 0, level}
                # when quantizing for real, multiply the row by 1/ level, and the bias by 1/ level
                #  scaling neurons will be inserted after the linear layer with a value of level.
        self.param_groups = self.base_optimizer.param_groups
        n_reg_groups = len(list(self.get_regularized_param_groups()))
        if n_reg_groups > 1:
            logger.warning(
                "Multiple regularized groups detected, this is maybe not supported yet."
            )

    def is_quantization_enabled(self):
        if self.state["steps"] <= self.step_start_reg:
            return False
        return True

    def get_regularized_param_groups(self):
        for group in self.base_optimizer.param_groups:
            if group.get("quant_bits", 16) < 16:
                yield group

    @torch.no_grad()
    def compute_factors_lsbq(self):
        """
        uses lsbq, copyright Apple MIT
        """
        #self.logger.info("COMPUTING FACTORS: LSQB")
        # self.row_levels_inverse.clear()
        
        factors_iter = iter(self.state["factors"])
        
        
        for group in self.get_regularized_param_groups():
            if group["quant_bits"] == 0:  # we call ternary 2 bit, they do 0 bit
                for x in group["params"]:
                    factor = lsbq_ternary(x.data.detach()).flatten()
                    is_any_nan = factor.isnan().any().item()
                    is_any_0 = (factor==0.).any().item()
                    if is_any_nan or is_any_0:
                        self.logger.info(f"{is_any_nan=} and {is_any_0=}")

                    next(factors_iter).copy_(factor)
                    # self.row_levels_inverse.append(
                    #     (third.abs() + two_thirds.abs()).detach()
                    # )
                    # print(x.shape)
        self.logger.info("END FACTORS COMPUTE")

    @torch.no_grad()
    def compute_factors_equisplit(self, percentile = 1/3):
        """

        """
        #self.logger.info("COMPUTING FACTORS: ")
        factors_iter = iter(self.state["factors"])
        for group in self.get_regularized_param_groups():
            if group["quant_bits"] == 0:
                for x in group["params"]:
                    factor = equisplit(x.data.detach(),percentile=percentile)
                    next(factors_iter).copy_(factor)
        # self.logger.info("END FACTORS COMPUTE")


    def set_final_beta(self):
        """
        sets all following betas to 1 and disables requires_grad on regParams
        """
        self.beta_schedule = itertools.repeat(1.0)
        for param in self.regularized_params:
            param.requires_grad = False # is will disable gradient descent, but this will most likely 
            # not change anything, because beta is now 1 and hard rounding gets used.

    def step(self: Self):
        """
        does not accept a closure yet!
        """
        self.state["steps"] += 1
        self.base_optimizer.step()
        if not self.is_quantization_enabled():
            return  # do stuff as normal
        else:
            # if self.steps == self.step_start_reg: # do this in train loop
            #    # enable scale:
            #    scale_lin_layers(self)
            pass  # continues with prox operator
        # print("STEPPING WITH PROX")
        # define map (per tensor later, or per channel, row etc.)
        # abs(param) > M map will always be needed
        # but based on beta <> 1 we should make different maps

        M = self.M
        gen = self.beta_schedule
        self.beta = next(gen)
        self.reg_function_(self.regularized_params,self.state["factors"],self.beta)



if __name__ == "__main__":
    n = 10
    k = (1/ n) ** 0.5
    x = torch.zeros((100,n))
    x.uniform_(-k,k)
    v = lsbq_ternary(x)
    print(v)
