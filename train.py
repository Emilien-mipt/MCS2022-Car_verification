import numpy as np
import torch
from tqdm import tqdm

from utils import AverageMeter, accuracy, warm_up_lr


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    epoch,
    device,
    scheduler=None,
):
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch: epoch number
    :param device:
    :param scheduler:
    :return: None
    """
    model.train()

    loss_stat = AverageMeter("Loss")
    top5_stat = AverageMeter("Prec@5")
    acc_stat = AverageMeter("Acc.")

    NUM_EPOCH_WARM_UP = config.train.n_epoch // 25
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP

    LR = config.train.learning_rate

    train_iter = tqdm(train_loader, desc="Train", dynamic_ncols=True, position=1)

    for step, (x, y) in enumerate(train_iter):
        if config.train.warmup:
            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                step + 1 <= NUM_BATCH_WARM_UP
            ):  # adjust LR for each training batch during warm up
                warm_up_lr(step + 1, NUM_BATCH_WARM_UP, LR, optimizer)

        num_of_samples = x.shape[0]

        optimizer.zero_grad()

        x = x.to(device).to(memory_format=torch.contiguous_format)
        y = y.to(device)
        out = model(x, y)
        loss = criterion(out, y)

        loss_stat.update(loss.detach().cpu().item(), num_of_samples)
        _, prec5 = accuracy(out.data, y, topk=(1, 5))
        top5_stat.update(prec5.data.item(), num_of_samples)

        loss.backward()
        optimizer.step()

        scores = torch.softmax(out, dim=1).detach().cpu().numpy()
        predict = np.argmax(scores, axis=1)
        gt = y.detach().cpu().numpy()

        acc = np.mean(gt == predict)
        acc_stat.update(acc, num_of_samples)

        if step % config.train.freq_vis == 0 and not step == 0:
            acc_val, acc_avg = acc_stat()
            loss_val, loss_avg = loss_stat()
            top5_val, top5_avg = top5_stat()
            print(
                "Epoch: {}; step: {}; loss: {:.4f}; acc: {:.8f}; prec@5: {:.8f}".format(
                    epoch, step, loss_avg, acc_avg, top5_avg
                )
            )

    if scheduler is not None:
        scheduler.step()

    acc_val, acc_avg = acc_stat()
    loss_val, loss_avg = loss_stat()
    top5_val, top5_avg = top5_stat()
    print(
        "Train process of epoch: {} is done; \n "
        "TRAIN_LOSS: {:.4f}; TRAIN_ACC: {:.8f}; \
        Prec@5: {:.8f}".format(
            epoch, loss_avg, acc_avg, top5_avg
        )
    )
    return loss_avg, acc_avg, top5_avg


def validation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    epoch,
    device,
):
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch:
    :param device:
    :return: float: avg acc
    """
    loss_stat = AverageMeter("Loss")
    top5_stat = AverageMeter("Prec@5")
    acc_stat = AverageMeter("Acc.")

    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader, desc="Val", dynamic_ncols=True, position=2)

        for step, (x, y) in enumerate(val_iter):
            num_of_samples = x.shape[0]

            x = x.to(device).to(memory_format=torch.contiguous_format)
            y = y.to(device)

            out = model(x, y)
            loss = criterion(out, y)

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            _, prec5 = accuracy(out.data, y, topk=(1, 5))
            top5_stat.update(prec5.data.item(), num_of_samples)

            scores = torch.softmax(out, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        top5_val, top5_avg = top5_stat()

        print(
            "Validation of epoch: {} is done; \n VAL_LOSS: {:.4f}; \
            VAL_ACC: {:.8f}; Prec@5: {:.8f}".format(
                epoch, loss_avg, acc_avg, top5_avg
            )
        )
        return loss_avg, acc_avg, top5_avg
