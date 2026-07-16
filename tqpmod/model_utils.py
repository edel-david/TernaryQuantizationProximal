import torch
import torch.nn as nn
from tqpmod.tqp_optimizer import equisplit


class ScaleLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        # print(self.scale.shape,x.shape)
        return x * self.scale


def transform_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        key: str
        new_state_dict[key.replace("_orig_mod.", "")] = value
    return new_state_dict

@torch.no_grad()
def rebalance_layers(
    model: nn.Module,
    param_dict: dict[nn.Parameter,dict],
    lr_schedule:torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim,
    dtype_target=torch.float32,
    dtype_calc=torch.float64,
    factor_function=equisplit,
    factor_function_args=(),
    factor_power = 2,
):
    for name, child in model.named_children():
        if isinstance(child,nn.Sequential) and (len(child) == 2) and isinstance(child[1],ScaleLayer):
            linear:nn.Linear | nn.Conv2d = child[0]
            scale:ScaleLayer = child[1]
            scale_shape = scale.scale.shape
            # if "conv1" in name:
            #     bn = model.bn1                
            # elif "conv2" in name:
            #     bn = model.bn2
            # else:
            #     # for the linear layer at the end:
            #     # bn = nn.BatchNorm2d(torch.numel(scale_shape)) # dummy batchnorm that gets destroyed at function close.
            #     pass
            # bn:nn.BatchNorm2d
            

            
            factors = factor_function(
                linear.weight.to(dtype_calc), *factor_function_args
            )
            factors = factors.reshape((-1, 1))#.to(dtype_target)

            if isinstance(linear, nn.Conv2d):
                factors = factors.reshape((*factors.shape, 1, 1))

            linear.weight.copy_(
                (linear.weight.detach().to(dtype_calc) / factors).to(dtype_target)
            )
            param_dict[linear.weight]["group"]["lr"] = param_dict[linear.weight]["group"]["lr"] * ((1 / factors.mean()) ** factor_power).to(
                        dtype_target
                    ).item()
            index_weight = param_dict[linear.weight]["index"]
            lr_schedule.base_lrs[index_weight]=lr_schedule.base_lrs[index_weight] * ((1 / factors.mean()) ** factor_power).to(
                        dtype_target
                    ).item()
            
            param_dict[linear.weight]["group"]["weight_decay"] = param_dict[linear.weight]["group"]["weight_decay"] * (factors.mean() ** factor_power).to(
                        dtype_target
                    ).item()
            if "momentum_buffer" in optimizer.base_optimizer.state[linear.weight].keys(): #SGDM
                optimizer.base_optimizer.state[linear.weight]["momentum_buffer"].mul_(factors.to(dtype_target))
            if "exp_avg" in optimizer.base_optimizer.state[linear.weight].keys(): #ADAM
                optimizer.base_optimizer.state[linear.weight]["exp_avg"].mul_(factors.to(dtype_target))
            if "exp_avg_sq" in optimizer.base_optimizer.state[linear.weight].keys():# ADAM
                optimizer.base_optimizer.state[linear.weight]["exp_avg_sq"].mul_((factors ** 2).to(dtype_target))
            if linear.bias is not None:
                if isinstance(linear, nn.Conv2d):
                    print("TODO: check if this bias is correct for conv2d bias")
                linear.bias.div_(factors.flatten())
                param_dict[linear.bias]["group"]["lr"] = param_dict[linear.bias]["group"]["lr"] * ((1 / factors.mean()) ** factor_power).to(
                        dtype_target
                    ).item()
                index_bias = param_dict[linear.bias]["index"]
                lr_schedule.base_lrs[index_bias]=lr_schedule.base_lrs[index_bias] * ((1 / factors.mean()) ** factor_power).to(
                        dtype_target
                    ).item()
                if "momentum_buffer" in optimizer.base_optimizer.state[linear.bias].keys():
                    optimizer.base_optimizer.state[linear.bias]["momentum_buffer"].mul_(factors.flatten().to(dtype_target))
                if "exp_avg" in optimizer.base_optimizer.state[linear.bias].keys():
                    optimizer.base_optimizer.state[linear.bias]["exp_avg"].mul_(factors.flatten().to(dtype_target))
                if "exp_avg_sq" in optimizer.base_optimizer.state[linear.bias].keys():
                    optimizer.base_optimizer.state[linear.bias]["exp_avg_sq"].mul_((factors.flatten() ** 2).to(dtype_target))

            scale.scale.mul_(factors.reshape(scale_shape).to(dtype_target))

            # BatchNorm scale:
            # stays the same

            # TODO: maybe also scale: scale lr.
        else:
            rebalance_layers(
                child,
                param_dict=param_dict,
                lr_schedule=lr_schedule,
                optimizer=optimizer,
                dtype_target=dtype_target,
                dtype_calc=dtype_calc,
                factor_function=factor_function,
                factor_function_args=factor_function_args,
                factor_power=factor_power,
            )


@torch.no_grad()
def inject_scale_layers(
    model: nn.Module,
    param_dicts_list: list,
    dtype_target=torch.float32,
    dtype_calc=torch.float64,
    factor_function=equisplit,
    factor_function_args=(),
    learning_rate=0.1,
    weight_decay=0.02,
    scale_grad = True,
    factor_power = 2
):
    for name, child in model.named_children():
        # Check for both Linear and Conv2d
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            # Create a sequence: Original Layer -> Test Layer

            factors = factor_function(
                child.weight.to(dtype_calc), *factor_function_args
            )
            factors = factors.reshape((-1, 1))

            scale_shape = (1, child.weight.shape[0])

            if isinstance(child, nn.Conv2d):
                scale_shape = (*scale_shape, 1, 1)  # append 1,1 to shape for conv
                factors = factors.reshape((*factors.shape, 1, 1))

            child.weight.copy_(
                ((child.weight.detach().to(dtype_calc) / factors).to(dtype_target)).to(
                    dtype_target
                )
            )
            if child.bias is not None:
                if isinstance(child, nn.Conv2d):
                    print("TODO: check if this bias is correct for conv2d bias")
                child.bias.div_(factors.flatten())
                param_dicts_list.append(  # **2 for sgd
                {
                    "params": [child.bias],
                    "weight_decay": 0,
                    "lr": (learning_rate * ((1 / factors.mean()) ** factor_power)).to(
                        dtype_target
                    ),
                }
            )
            scale = ScaleLayer(scale_shape).to(child.weight.device)
            scale.scale.requires_grad=scale_grad
            scale.scale.mul_(factors.to(dtype_target).reshape(scale_shape))
            new_block = nn.Sequential(child, scale)
            
            # Replace the attribute on the model
            setattr(model, name, new_block)
            param_dicts_list.append(  # **2 for sgd
                {
                    "params": [child.weight],
                    "quant_bits": 0,
                    "weight_decay": (weight_decay * (factors.mean() ** factor_power)).to(dtype_target),
                    "lr": (learning_rate * ((1 / factors.mean()) ** factor_power)).to(
                        dtype_target
                    ),
                }
            )
            

        else:
            # Recurse for nested modules (like Bottlenecks in ResNet)

            inject_scale_layers(
                child,
                param_dicts_list,
                dtype_target=dtype_target,
                dtype_calc=dtype_calc,
                factor_function=factor_function,
                factor_function_args=factor_function_args,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                scale_grad=scale_grad,
                factor_power=factor_power,
            )


@torch.no_grad()
def inject_scale_layers_no_factor(
    model: nn.Module,
    param_dicts_list: list,
    learning_rate=0.1,
    weight_decay=2e-4,
    scale_grad = True,
):
    for name, child in model.named_children():
        # Check for both Linear and Conv2d
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            # Create a sequence: Original Layer -> Test Layer

            scale_shape = (1, child.weight.shape[0])

            if isinstance(child, nn.Conv2d):
                scale_shape = (*scale_shape, 1, 1)  # append 1,1 to shape for conv
            if child.bias is not None:
                if isinstance(child, nn.Conv2d):
                    print("TODO: check if this bias is correct for conv2d bias")
                param_dicts_list.append(  
                {
                    "params": [child.bias],
                    "weight_decay": 0,
                    "lr": (learning_rate),
                }
            )
            scale = ScaleLayer(scale_shape).to(child.weight.device)
            scale.scale.requires_grad=scale_grad
            new_block = nn.Sequential(child, scale)
            
            # Replace the attribute on the model
            setattr(model, name, new_block)
            param_dicts_list.append( 
                {
                    "params": [child.weight],
                    "quant_bits": 0,
                    "weight_decay": (weight_decay),
                    "lr": (learning_rate),
                }
            )
            

        else:
            # Recurse for nested modules (like Bottlenecks in ResNet)

            inject_scale_layers_no_factor(
                child,
                param_dicts_list,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                scale_grad=scale_grad
            )


@torch.no_grad()
def inject_test_layers_quantized_model(
    model: nn.Module,
    parent_name="model_model",
    dtype_target=torch.float32,
    dtype_calc=torch.float64,
):
    for name, child in [x for x in model.named_children()]:
        full_name = f"{parent_name}_{name}"
        # Check for both Linear and Conv2d
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            print(full_name)
            # if full_name in ["base_model_conv1","base_model_linear"]:
            # if full_name in ["base_model_conv1"]:
            # if full_name in ["base_model_linear"]:
            #
            #   print("skipping: ",full_name)
            #   continue
            new_child = child.weight.detach()
            child_shape = new_child.shape
            new_child = new_child.reshape((new_child.shape[0], -1))
            child_max_per_row = torch.amax(new_child, (1,)).reshape(
                (1, new_child.shape[0])
            )
            new_child = torch.where(new_child > 0.0, 1.0, new_child)
            new_child = torch.where(new_child < 0.0, -1.0, new_child)

            child.weight.copy_(
                ((new_child).reshape(child_shape).to(dtype_target)).to(dtype_target)
            )
            if child.bias is not None:
                child.bias.mul_(1 / child_max_per_row.flatten())
            # child.weight.round_()
            scale_shape = (1, new_child.shape[0])
            if isinstance(child, nn.Conv2d):
                scale_shape = (*scale_shape, 1, 1)
            scale = ScaleLayer(scale_shape).to(child.weight.device)
            # print(scale_shape)
            scale.scale.copy_(child_max_per_row.to(dtype_target).reshape(scale_shape))
            new_block = nn.Sequential(child, scale)

            # Replace the attribute on the model
            setattr(model, name, new_block)
        else:
            # Recurse for nested modules (like Bottlenecks in ResNet)
            inject_test_layers_quantized_model(
                child, full_name, dtype_target=dtype_target, dtype_calc=dtype_calc
            )


def inject_test_layers_unrescale(model: nn.Module):
    dtype_target = torch.float32
    dtype_calc = torch.float64
    for name, child in model.named_children():
        # Check for both Linear and Conv2d
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            factor_shape_unsqueeze_dims = len(child.weight.shape) - 2  #
            scale_shape = (child.weight.shape[0],)
            for _ in range(factor_shape_unsqueeze_dims):
                scale_shape = (*scale_shape, 1)
            scale = ScaleLayer(scale_shape).to(child.weight.device)
            new_block = nn.Sequential(child, scale)

            # Replace the attribute on the model
            setattr(model, name, new_block)
        else:
            # Recurse for nested modules (like Bottlenecks in ResNet)
            inject_test_layers_unrescale(child)
