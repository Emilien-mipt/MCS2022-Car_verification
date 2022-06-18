import numpy as np
import torch
from tqdm import tqdm

from losses.circle_loss import convert_label_to_similarity
from utils import AverageMeter


def train(
    model: torch.nn.Module,
    criterion,
    head_criterion,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config,
    epoch,
):
    """
    :param model: model architecture
    :param criterion:
    :param head_criterion:
    :param train_loader: dataloader for batch generation
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch:
    :return: tuple(float, float): Avg loss and avg accuracy
    """
    model.train()

    loss_stat = AverageMeter("Loss")
    acc_stat = AverageMeter("Acc.")

    train_iter = tqdm(train_loader, desc="Train", dynamic_ncols=True, position=1)

    train_dataset_size = len(train_loader.dataset)

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = (
        round(train_dataset_size / config.dataset.batch_size) * config.train.warm_epoch
    )  # first 5 epoch

    for step, (inputs, labels) in enumerate(train_iter):
        now_batch_size = inputs.shape[0]
        inputs = inputs.cuda().to(memory_format=torch.contiguous_format)
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)

        head = config.model.head

        if head:
            logits, ff = outputs
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            loss = criterion(logits, labels)
            _, preds = torch.max(logits.data, 1)
            if (
                (head == "arcface")
                or (head == "cosface")
                or (head == "instance")
                or (head == "sphere")
            ):
                loss += head_criterion(ff, labels) / now_batch_size
            elif (head == "lifted") or (head == "contrast"):
                loss += head_criterion(ff, labels)
            elif head == "circle":
                loss += (
                    head_criterion(*convert_label_to_similarity(ff, labels))
                    / now_batch_size
                )
            else:
                raise ValueError("Wrong head! Please check the config file")
        else:
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

        # backward
        if epoch < config.train.warm_epoch:
            warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
            loss = loss * warm_up

        loss.backward()
        optimizer.step()

        loss_stat.update(loss.detach().cpu().item(), now_batch_size)

        gt = labels.detach().cpu().numpy()
        acc = np.mean(gt == preds)
        acc_stat.update(acc, now_batch_size)

        if step % config.train.freq_vis == 0 and not step == 0:
            acc_val, acc_avg = acc_stat()
            loss_val, loss_avg = loss_stat()
            print(
                "Epoch: {}; step: {}; loss: {:.4f}; acc: {:.2f}".format(
                    epoch, step, loss_avg, acc_avg
                )
            )

    acc_val, acc_avg = acc_stat()
    loss_val, loss_avg = loss_stat()
    print(
        "Train process of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}".format(
            epoch, loss_avg, acc_avg
        )
    )
    return loss_avg, acc_avg


def validation(
    model: torch.nn.Module,
    criterion,
    head_criterion,
    val_loader: torch.utils.data.DataLoader,
    config,
    epoch,
):
    """
    Model validation function for one epoch
    :param model: model architecture
    :param criterion:
    :param head_criterion:
    :param val_loader: dataloader for batch generation
    :param config:
    :param epoch:
    :return: tuple(float, float): Avg loss and avg accuracy
    """
    loss_stat = AverageMeter("Loss")
    acc_stat = AverageMeter("Acc.")

    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader, desc="Val", dynamic_ncols=True, position=2)

        for step, (inputs, labels) in enumerate(val_iter):
            now_batch_size = inputs.shape[0]
            inputs = inputs.cuda().to(memory_format=torch.contiguous_format)
            labels = labels.cuda()

            outputs = model(inputs)

            head = config.model.head

            if head:
                logits, ff = outputs
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
                loss = criterion(logits, labels)
                _, preds = torch.max(logits.data, 1)
                if (
                    (head == "arcface")
                    or (head == "cosface")
                    or (head == "instance")
                    or (head == "sphere")
                ):
                    loss += head_criterion(ff, labels) / now_batch_size
                elif (head == "lifted") or (head == "contrast"):
                    loss += head_criterion(ff, labels)
                elif head == "circle":
                    loss += (
                        head_criterion(*convert_label_to_similarity(ff, labels))
                        / now_batch_size
                    )
                else:
                    raise ValueError("Wrong head! Please check the config file")
            else:
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

            loss_stat.update(loss.detach().cpu().item(), now_batch_size)

            gt = labels.detach().cpu().numpy()
            acc = np.mean(gt == preds)
            acc_stat.update(acc, now_batch_size)

            if step % config.train.freq_vis == 0 and not step == 0:
                acc_val, acc_avg = acc_stat()
                loss_val, loss_avg = loss_stat()
                print(
                    "Epoch: {}; step: {}; loss: {:.4f}; acc: {:.2f}".format(
                        epoch, step, loss_avg, acc_avg
                    )
                )

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        print(
            "Train process of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}".format(
                epoch, loss_avg, acc_avg
            )
        )
        return loss_avg, acc_avg
