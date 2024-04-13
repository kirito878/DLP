import argparse
import oxford_pet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import models.unet as unet
import evaluate
import utils 
import os
def predict_mask(net, data, device):
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
    return prediction
def get_test_name(path):
    path = os.path.join(path,"annotations/test.txt")
    with open(path) as f:
        split_data = f.read().strip("\n").split("\n")
    filenames = [x.split(" ")[0] for x in split_data]
    return filenames
def inference(args):
    model_path = args.model
    data_path = args.data_path
    batch_size = args.batch_size
    device = args.device
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    test_data = oxford_pet.load_dataset(data_path, 'test')
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = torch.load(model_path).to(device)
    dice_score = evaluate.evaluate(model,test_loader,device)
    predicts =  predict_mask(model,test_loader,device)
    filenames = get_test_name(data_path)
    for index,i in enumerate(predicts):
        name = filenames[index]
        print(name)
        utils.visualize(save_path,name,i)
    print(f"dice score: {dice_score}")

def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth',
                        help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=1, help='batch size')
    parser.add_argument("--device", type=str,
                        default='cuda', help='model device')
    parser.add_argument("--save_path",type=str,help='path to save the predict mask')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    inference(args)
    # assert False, "Not implemented yet!"
