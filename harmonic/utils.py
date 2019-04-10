import os
import os.path

import numpy as np
import torch
import torch.nn.functional as F

from harmonic import AddSine, Embedding

def print_percent(percent):
    print(str(percent * 5) + "%")


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def recursive_search(net, class_to_find):
    result = []
    for child in net.children():
        if isinstance(child, class_to_find):
            result += [child]
        else:
            result += recursive_search(child, class_to_find)
    return result


def disable_gradients_for_class(net, class_to_find, requires_grad=False):
    instances = recursive_search(net, class_to_find)
    for layer in instances:
        for param in layer.parameters():
            param.requires_grad = requires_grad
    return instances


def train(net, dataloader, emb, caption, device, n_epoch=10, unfreeze_epoch=0, lr=1e-5,
          verbose=True, ignore_bg=False, floss=F.l1_loss):
    """
    
    :param net: neural network to be trained
    :param dataloader: dataset
    :param emb: embedding system (see embeddings)
    :param caption: the name of files to store temporaly information
    :param device: name of the device 
    :param n_epoch: number of epoch to train
    :param unfreeze_epoch: 
    :param lr: learning rate of the optimizer
    :param verbose: 
    :param ignore_bg: if the we don't use 
    :param floss: loss to calculate
    :return: trained network and errors
    """
    print('learing started')

    assert isinstance(dataloader, torch.utils.DataLoader), "Dataloader is required"
    assert isinstance(emb, Embedding), "Object of class Embedding is required"
    net.train()
    # We don't want to optimize w.r.t. to our embedding params cause they are fixed
    disable_gradients_for_class(net, AddSine, requires_grad=False)
    errors = []

    percent_old = 0
    if verbose:
        print_percent(percent_old)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.children().next().freeze_encoder(False)

    if caption:
        if os.path.isfile(caption + '_ckp.t7'):
            checkpoint = torch.load(caption + '_ckp.t7')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            if lr > 0.:
                set_lr(optimizer, lr)
            if os.path.isfile(caption + '_log.txt'):
                data = np.loadtxt(caption + '_log.txt')
                if data.shape[0] > 0:
                    errors = list(data)

    ndata = len(dataloader)
    e = 0
    for e in range(n_epoch):
        print('epoch:', e, ' of ', n_epoch)
        if e == unfreeze_epoch:
            net.children().next().freeze_encoder(True)

        for i, data in enumerate(dataloader):

            if 3 == len(data):
                x, y, z = data
            elif 2 == len(data):
                x, y = data
            else:
                raise ValueError("unexpected data format")

            optimizer.zero_grad()

            vx = x.float().to(device)

            optimizer.zero_grad()

            vy, w = emb(y[:, 0].numpy())

            res = net(vx)

            if ignore_bg:
                f1 = floss(res, vy.detach(), reduce=False)
                if emb.weights_norm is None:
                    if torch.sum(y[:, 0] > 0) > 0:
                        loss = torch.mean(torch.mean(f1, dim=1)[y[:, 0] > 0])
                else:
                    if torch.sum(y[:, 0] > 0) > 0:
                        loss = torch.mean((w.detach() * torch.mean(f1, dim=1))[y[:, 0] > 0])
            else:
                if emb.weights_norm is None:
                    loss = floss(res, vy.detach())
                else:
                    f1 = floss(res, vy.detach(), reduce=False)
                    loss = torch.mean(w.detach() * torch.mean(f1, dim=1))

            errors.append([loss.item()])
            loss.backward()
            optimizer.step()

            if 0 == i % 500:
                np.savetxt(caption + '_log.txt', errors)
                torch.save({
                    'epoch': e,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, caption + '_ckp.t7')

            if verbose:
                percent = int(float(i) / float(ndata) * 20.)
                if percent_old != percent:
                    percent_old = percent
                    print_percent(percent)

        if verbose:
            print_percent(20)

    torch.save({
        'epoch': e,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, caption + '_ckp.t7')

    torch.save(net.state_dict(), caption + '.t7')
    np.savetxt(caption + '_log.txt', errors)
    return net, errors
