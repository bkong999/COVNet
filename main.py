import os
import numpy as np
import time

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dataset
import model
import config
import utils


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, train_loader, epoch, num_epochs, optimizer, writer,
                current_lr, log_every=100):
    n_classes = model.n_classes
    metric = torch.nn.CrossEntropyLoss()

    y_probs = np.zeros((0, n_classes), np.float)
    losses, y_trues = [], []
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    for i, (image, label, case_id) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues.append(label.item())

        metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
        n_iter = epoch * len(train_loader) + i
        writer.add_scalar('Train/Loss', loss_value, n_iter)

        if (i % log_every == 0) & (i > 0):
            utils.print_progress(epoch + 1, num_epochs, i, len(train_loader),
                                 np.mean(losses), current_lr, metric_collects)

    train_loss_epoch = np.round(np.mean(losses), 4)
    return train_loss_epoch, metric_collects


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr,
                   log_every=20):
    n_classes = model.n_classes
    metric = torch.nn.CrossEntropyLoss()

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    y_probs = np.zeros((0, n_classes), np.float)
    losses, y_trues = [], []

    for i, (image, label, case_id) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues.append(label.item())

        metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)

        n_iter = epoch * len(val_loader) + i
        writer.add_scalar('Val/Loss', loss_value, n_iter)

        if (i % log_every == 0) & (i > 0):
            prefix = '*Val|'
            utils.print_progress(epoch + 1, num_epochs, i, len(val_loader),
                                 np.mean(losses), current_lr, metric_collects,
                                 prefix=prefix)

    val_loss_epoch = np.round(np.mean(losses), 4)
    return val_loss_epoch, metric_collects


def main(args):
    """Main function for the training pipeline

    :args: commandlien arguments
    :returns: None

    """
    ##########################################################################
    #                             Basic settings                             #
    ##########################################################################
    exp_dir = 'experiments'
    log_dir = os.path.join(exp_dir, 'logs')
    model_dir = os.path.join(exp_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    ##########################################################################
    #  Define all the necessary variables for model training and evaluation  #
    ##########################################################################
    writer = SummaryWriter(log_dir)
    train_dataset = dataset.NCovDataset('data/', stage='train')
    weights = train_dataset.make_weights_for_balanced_classes()
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(train_dataset.case_ids))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=20,
        drop_last=False, sampler=sampler)

    val_dataset = dataset.NCovDataset('data/', stage='val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=11,
        drop_last=False)

    cov_net = model.COVNet(n_classes=args.n_classes)
    if torch.cuda.is_available():
        cov_net = cov_net.cuda()
    optimizer = optim.Adam(cov_net.parameters(), lr=args.lr, weight_decay=0.1)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)

    best_val_loss = float('inf')
    best_val_accu = float(0)

    iteration_change_loss = 0
    t_start_training = time.time()
    ##########################################################################
    #                           Main training loop                           #
    ##########################################################################
    for epoch in range(args.epochs):
        current_lr = get_lr(optimizer)
        t_start = time.time()

        ############################################################
        #  The actual training and validation step for each epoch  #
        ############################################################
        train_loss, train_metric = train_model(
            cov_net, train_loader, epoch, args.epochs, optimizer, writer,
            current_lr, args.log_every)

        with torch.no_grad():
            val_loss, val_metric = evaluate_model(
                cov_net, val_loader, epoch, args.epochs, writer, current_lr)

        ##############################
        #  Adjust the learning rate  #
        ##############################
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        utils.print_epoch_progress(train_loss, val_loss, delta, train_metric,
                                   val_metric)
        iteration_change_loss += 1
        print('-' * 30)

        train_acc, val_acc = train_metric['accuracy'], val_metric['accuracy']
        file_name = ('train_acc_{}_val_acc_{}_epoch_{}.pth'.
                     format(train_acc, val_acc, epoch))
        torch.save(cov_net, os.path.join(model_dir, file_name))

        if val_acc > best_val_accu:
            best_val_accu = val_acc
            if bool(args.save_model):
                torch.save(cov_net, os.path.join(model_dir, 'best.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == args.patience:
            print(('Early stopping after {0} iterations without the decrease ' +
                  'of the val loss').format(iteration_change_loss))
            break
    t_end_training = time.time()
    print('training took {}s'.
          format(t_end_training - t_start_training))


if __name__ == "__main__":
    args = config.parse_arguments()
    main(args)
