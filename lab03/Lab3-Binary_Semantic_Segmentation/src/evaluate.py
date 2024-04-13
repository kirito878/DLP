import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import ipdb
def evaluate(net, data, device):
    # implement the evaluation function here
    net.to(device)
    net.eval()
    prediction = []
    ground_truth = []
    with torch.no_grad():
        for i, sample_data in enumerate(data, 0):
            x_data = sample_data['image'].to(device)
            y_label = sample_data['mask'].to(device)
            y_pred = net(x_data)
            mask = (torch.sigmoid(y_pred)> 0.5).cpu().numpy().astype(int)
            # ipdb.set_trace()
            prediction.extend(mask)
            ground_truth.extend(y_label.cpu().numpy().astype(int))
    dice_score = utils.mean_dice_score(prediction,ground_truth)
    return dice_score

