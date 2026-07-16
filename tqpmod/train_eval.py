import torch
from tqdm import tqdm


# Training function
def train_epoch(model, loader, criterion, optimizer, scaler, device, trace=False, epoch = None,amp=False):
    model.train()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0  # cpu side, because we add ints to it (from shape) which are already on cpu and dont need to sync with the gpu

    pbar = tqdm(loader, desc="Training")
    optimizer.base_optimizer.zero_grad()
    for images, labels in pbar:

        torch.compiler.cudagraph_mark_step_begin()
        images: torch.Tensor
        labels: torch.Tensor
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        # Mixed precision training
        with torch.amp.autocast("cuda",enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.base_optimizer.zero_grad()
        with torch.no_grad():
            running_loss += loss.detach()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().detach()
            total += labels.size(0)


    avg_loss = running_loss.item() / len(loader)
    avg_accuracy = 100.0 * correct.item() / total
    #logger.info(f"loss {avg_loss:.4f}   acc {avg_accuracy:.2f}%")
    return avg_loss, avg_accuracy


# Evaluation function
@torch.no_grad()
def evaluate(model, loader, criterion, device,transform_train_x_dtype = torch.float32, amp=False):
    model.eval()
    running_loss = torch.zeros((), device=device, requires_grad=False)
    correct = torch.zeros((), device=device, requires_grad=False)
    total = 0

    pbar = tqdm(loader, desc="Evaluating")
    for images, labels in pbar:
        torch.compiler.cudagraph_mark_step_begin() # probably not needed
        images: torch.Tensor
        labels: torch.Tensor
        images, labels = images.to(device,transform_train_x_dtype, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        with torch.amp.autocast("cuda",enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.detach()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().detach()
        total += labels.size(0)

    avg_loss = running_loss.item() / len(loader)
    test_accuracy = 100.0 * correct.item() / total
    return avg_loss, test_accuracy

from model import resnet
from tqpmod.model_utils import transform_state_dict,ScaleLayer,inject_scale_layers
@torch.no_grad()
def fake_quant_eval(model, loader, criterion, device):

    model_clone :resnet.ResNet=resnet.__dict__["resnet20"]()
    model_clone.load_state_dict(transform_state_dict( model.state_dict()))

    inject_scale_layers(model_clone)
    model_clone.eval()
    model_clone.to(device)
    test_loss, test_acc = evaluate(model_clone,loader,criterion,device)
    return test_loss, test_acc