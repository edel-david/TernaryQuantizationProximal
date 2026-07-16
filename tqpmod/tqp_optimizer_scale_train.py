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


def tanh_list(params, factors, beta):
    for param, factor in zip(params, factors):
        # param.copy_()
        raise NotImplementedError()


def piece_quad_prox_scale_layer(params, beta):
    if beta < 1.0:
        apply_prox_to_list_optim_scale_layer(
            params,
            beta,
        )
    else:
        apply_prox_hard_rounding_list_scale_layer(params, beta=beta)


@torch.compile()
@torch.no_grad()
def apply_prox_hard_rounding_list_scale_layer(params: torch.Tensor, beta):
    for param in params:
        param.clip_(-1.0, 1.0).round_()
        # param.copy_(apply_hard_rounding_ternary_scale_layer(param.data).detach())


@torch.compile()
@torch.no_grad()
def apply_prox_to_list_optim_scale_layer(params, beta):
    """
    assume beta < 1 and M = 1
    """
    for param in params:
        param.copy_(apply_prox_same_beta_optim_scale_layer(param.data, beta).detach())


@torch.compile(dynamic=False)
@torch.no_grad()
def apply_prox_same_beta_optim_scale_layer(theta_old: torch.Tensor, beta: torch.Tensor):
    # factor replaces the one
    # this means that factor * one = factor, which is where the slope will be 0 after the transformation
    """
    assume beta < 1 and M = 1
    will handle every tensor row-wise, so (A,B,C) will be treated as (A,B*C)
    """

    theta_new = theta_old
    a0 = beta / 2
    abs_theta = abs(theta_new)
    y = torch.where(abs_theta < a0, 0, abs_theta / (1 - beta) - (1 / (1 - beta) * a0))
    # y[abs_theta>(1-a0)]=1
    y = torch.where(abs_theta > (1 - a0), 1, y)
    y = torch.where(
        abs_theta > 1 + a0,
        abs_theta / (1 + beta) + (1 - ((1 + beta / 2) / (1 + beta))),
        y,
    )
    return y * torch.sign(theta_new)


@torch.compile()
@torch.no_grad()
def apply_hard_rounding_ternary_scale_layer(theta_old: torch.Tensor):
    theta_new = theta_old.clip(-1.0, 1.0).round()
    return theta_new


# @torch.compile()
@torch.no_grad()
def lsbq_ternary(x: torch.Tensor) -> torch.Tensor:
    x = x.flatten(1)
    v_cands = (
        x.abs().sort(dim=1).values
    )  # dim =1 is equal to dim = -1 , flatten conv to (x,-1)
    cumsum = v_cands.cumsum(dim=1)
    cumsum, total_sum = cumsum[:, 1:-1], cumsum[:, -1:]
    counts = torch.arange(1, x.size(dim=1), device=x.device)
    counts_r2l = counts[:-1].flip((-1,))
    cmean_r2l = (total_sum - cumsum).div_(counts_r2l.mul_(2))
    v_cands, v_cands2 = v_cands[:, 1:-1], v_cands[:, 2:]

    mask = (v_cands <= cmean_r2l).logical_and_(v_cands2 >= cmean_r2l)
    optimal_v = x.mean(dim=1, keepdim=True).div_(2)
    row_invalid = optimal_v < x.min(dim=1, keepdim=True).values
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
    r = r.sub(v * binary_sign(r))
    r = r.sub(v * binary_sign(r))
    costs = r.norm(dim=-1)
    indices = costs.argmin(dim=-1, keepdim=True)
    v_best = v_cands.gather(1, indices)
    v_best = v_best.flatten()
    v_best = torch.where(v_best == 0.0, x.abs().mean(1) * 0.7, v_best)
    return v_best * 2


# TODO: test if this can be compiled
@torch.no_grad()
def equisplit(x, percentile=1 / 3):
    """
    calcualtes factor based on quantiles
    sorts the matrix in each row
    takes the percenile * N th element and the (1- percentiles) * N th element and retuns their mean of abs


    x: at least 2 dim tensor
    percentile:
    """
    x = x.flatten(1)
    x_sorted, _ = x.detach().sort(1)  # now per row scaling
    x_sorted: torch.Tensor
    third = x_sorted[:, int(x_sorted.shape[1] * percentile), None]  # None for keepdim
    two_thirds = x_sorted[:, int(x_sorted.shape[1] * (1 - percentile)), None]
    equi_spit = third.abs() + two_thirds.abs()
    return equi_spit


class TQPS(Optimizer):
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
        beta: float = 0,
        device=torch.device("cuda:0"),
        logger=logging.getLogger(__name__),
        reg_function_=piece_quad_prox_scale_layer,
        n_reg_warmup_epochs = 10
    ):
        self.n_reg_warmup_epochs = n_reg_warmup_epochs
        self.reg_function_ = reg_function_
        self.logger = logger
        super().__init__(
            [{"params": []}], {"lr": base_optimizer.defaults["lr"]}
        )  # NOTE  check if params from base_optimizer should be placed here
        self.base_optimizer = base_optimizer
        self.beta = torch.tensor(
            1.0, requires_grad=False, device=device
        )  # gets overwritten below
        self.beta.mul_(beta)
        self.M = torch.tensor([1])
        self.state["steps"] = torch.zeros(
            (), requires_grad=False, device=torch.device("cpu")
        )
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = reg_wait_epochs
        self.total_steps = (regularization_epochs + reg_wait_epochs) * steps_per_epoch
        self.total_reg_steps = regularization_epochs * steps_per_epoch
        self.step_start_reg = reg_wait_epochs * steps_per_epoch

        # self.beta = next(self.beta_schedule)
        self.regularized_params: list[torch.Tensor] = []
        for group in self.get_regularized_param_groups():
            self.regularized_params.extend(
                group["params"]
            )  # append all reg parms to list for efficient
        #     # iteration later in compiled functions
        #     for param in group["params"]:
        #         # lets hope the ordering of this stays the same
        #         self.state["factors"].append(torch.ones((param.shape[0],1), device=device))
        # self.row_levels_inverse.append(
        #     torch.ones((param.shape[0],), device=device)
        # )  # this is a inverse_levels tensor,
        # matricies get multiplied x * W^T, so the first dim of W is the amount of rows.
        # so for each matrix (for now only 2d weights), this will give each row( the weights for one neuron) a level.
        # The Quantization targets for this row will be {-level, 0, level}
        # when quantizing for real, multiply the row by 1/ level, and the bias by 1/ level
        #  scaling neurons will be inserted after the linear layer with a value of level.
        self.param_groups = self.base_optimizer.param_groups

    def is_quantization_enabled(self):
        if self.state["steps"] <= self.step_start_reg:  # <= because we increment in
            # step before calling into here
            return False
        return True

    def get_regularized_param_groups(self):
        for group in self.base_optimizer.param_groups:
            if group.get("quant_bits", 16) < 16:
                yield group

    def set_final_beta(self):
        """
        sets all following betas to 1 and disables requires_grad on regParams
        """
        self.beta_schedule = itertools.repeat(1.0)
        self.beta = 1.0
        for param in self.regularized_params:
            param.requires_grad = (
                False  # is will disable gradient descent, but this will most likely
            )
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
            pass  # continues with prox operator
        # print("STEPPING WITH PROX")
        # define map (per tensor later, or per channel, row etc.)
        # abs(param) > M map will always be needed
        # but based on beta <> 1 we should make different maps
        reg_step = self.state["steps"] - self.step_start_reg
        warmup_factor = min(1, (reg_step / self.steps_per_epoch) / self.n_reg_warmup_epochs) # assumes 10 epochs for beta warmup are available
        # does increase local performance, but probably not global (later in training) performance
        self.reg_function_(self.regularized_params, self.beta * warmup_factor)


if __name__ == "__main__":
    n = 10
    k = (1 / n) ** 0.5
    x = torch.zeros((100, n))
    x.uniform_(-k, k)
    v = lsbq_ternary(x)
    print(v)
