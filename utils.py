import dataloader
from torch.utils.data import DataLoader

import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm

import numpy as np
from torch.autograd import Variable
import os

# for mixup, reference: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=0.8, device = None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device is not None:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# for mixup, reference: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_criterion(loss_fn, pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)


def train(model, args):
    # set training hyper-parameter
    lr = args["lr"]
    decay = args["decay"]
    secheduling = args["scheduling"]
    lr_decay = args["lr_decay"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    is_mixup = args["mixup"]
    device = args["device"]
    model_type = args["model"]
    alpha = args["alpha"]
    
    if args["finetune"]:
        # finetuned the feature extractor if using pre-trained model 
        finetune_lr = lr/10
        finetune_decay = decay/10
    else:
        finetune_lr = lr
        finetune_decay = decay

    if args["freeze"]:
        for param in list(model.Extractor.parameters())[:-3]:  # finetune the last 3 layers of the pre-trained model and freeze the remaining one
            param.requires_grad = False

    # prepare data and dataloader
    trainset = dataloader.CUB200(is_train=True)
    train_loader = DataLoader(trainset, batch_size, shuffle=True)

    # setup training 
    finetune_optim = optim.Adam(filter(lambda p: p.requires_grad, model.Extractor.parameters()), lr=finetune_lr, weight_decay=finetune_decay)  #filter out the freeze part
    if model_type.split("_")[0] == "resnet":
        optimizer = optim.Adam(list(model.avgpool.parameters())+list(model.fc.parameters()), lr=lr, weight_decay=decay)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=decay)

    finetune_scheduler = torch.optim.lr_scheduler.MultiStepLR(finetune_optim, secheduling, gamma=lr_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, secheduling, gamma=lr_decay)

    loss_fn = nn.CrossEntropyLoss()

    # model training
    epoch_loss = []
    epoch_acc = []
    for epoch in range(epochs):
        if args["mixup"]:
            
            if epoch < 5 or (epoch+1) > (epochs-5):
                is_mixup = False
            else:
                is_mixup = True
            
        prog_bar = tqdm(train_loader)
        _loss, _acc, num_data, iter = 0., 0., 0, 1
        
        model.train()
        for i, (inputs, targets) in enumerate(prog_bar, start=1):
            iter += 1
            inputs, targets = inputs.to(device), targets.to(device)

            finetune_optim.zero_grad()
            optimizer.zero_grad()

            if is_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, device)
                inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
                output = model(inputs)

                loss = mixup_criterion(loss_fn, output, targets_a, targets_b, lam)
            else:
                output= model(inputs)
                loss = loss_fn(output, targets)

            _loss += loss.item()
            
            loss.backward()
            
            finetune_optim.step()
            optimizer.step()

            _, pred = output.max(1)

            if is_mixup:
                _acc += (lam * pred.eq(targets_a).sum().float().item()
                            + (1 - lam) * pred.eq(targets_b).sum().float()).item()
            else:
                _acc += pred.eq(targets).sum().item()
            num_data += len(inputs)

            prog_bar.set_description(
                    "Epoch {}/{} => Total loss: {}, Acc: {}".format(
                        epoch + 1, args["epochs"],
                        round(_loss / i, 3),
                        round(_acc / num_data, 3)
                    )
                )
        finetune_scheduler.step()
        scheduler.step()
        epoch_loss.append(round(_loss/iter, 3))
        epoch_acc.append(round(_acc / num_data, 3))
    
    return model, epoch_loss, epoch_acc

def test(model, device):
    testset = dataloader.CUB200(is_train=False)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    test_prog_bar = tqdm(test_loader)
    classes_dict = dataloader.class_name_list()

    model.eval()
    answer = []
    with torch.no_grad():
        for i, inputs in enumerate(test_prog_bar):
            inputs = inputs.to(device)
            _, pred = model(inputs).max(1)
            answer.append([testset.imgs_dir[i], classes_dict[pred]])
    save = "answer"
    np.savetxt(os.path.join("./answer/" + save + ".txt"), answer, fmt='%s')