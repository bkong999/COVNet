import os
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

import model
import config
import dataset
import utils


def evaluate_model(model, val_loader):
    metric = torch.nn.CrossEntropyLoss()
    model.eval()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    y_probs = np.zeros((0, 3), np.float)
    losses, y_trues = [], []

    for i, (image, label, case_id) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1).detach().cpu().numpy()

        y_probs = np.concatenate([y_probs, y_prob])
        y_trues.append(label.item())
    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
    val_loss_epoch = np.mean(losses)
    return val_loss_epoch, metric_collects


def main(args):
    """Main function for the testing pipeline

    :args: commandline arguments
    :returns: None

    """
    ##########################################################################
    #                             Basic settings                             #
    ##########################################################################
    exp_dir = 'experiments'
    model_dir = os.path.join(exp_dir, 'models')
    model_file = os.path.join(model_dir, 'best.pth')
    val_dataset = dataset.NCovDataset('data/', stage='val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=11,
        drop_last=False)

    cov_net = model.COVNet(n_classes=3)
    if torch.cuda.is_available():
        cov_net.cuda()

    state = torch.load(model_file)
    cov_net.load_state_dict(state.state_dict())

    with torch.no_grad():
        val_loss, metric_collects = evaluate_model(cov_net, val_loader)
    prefix = '******Evaluate******'
    utils.print_progress(mean_loss=val_loss, metric_collects=metric_collects,
                         prefix=prefix)


if __name__ == "__main__":
    args = config.parse_arguments()
    main(args)
