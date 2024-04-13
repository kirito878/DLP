import argparse
import oxford_pet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import models.unet as unet
import evaluate
from torch.optim.lr_scheduler import StepLR
import models.resnet34_unet as resnet34_unet

def train(args):
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    device = args.device
    print(data_path, epochs, batch_size, lr, device)
    train_data = oxford_pet.load_dataset(data_path, 'train')
    valid_data = oxford_pet.load_dataset(data_path, 'valid')
    print(len(train_data))
    print(len(valid_data))
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False)
    # unet_model = unet.Unet_model().to(device)
    # loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(unet_model.parameters(), lr=lr)
    # best_score = 0
    # for epoch in range(1, epochs+1):
    #     total_loss = 0
    #     for i, data in enumerate(train_loader, 0):
    #         x_data = data['image'].to(device)
    #         y_label = data['mask'].to(device)
    #         y_pred = unet_model(x_data)
    #         loss = loss_fn(y_pred, y_label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item() * y_label.shape[0]
    #     total_loss /= len(train_loader)
    #     train_dice_score = evaluate.evaluate(unet_model, train_loader, device)
    #     vaild_dice_score = evaluate.evaluate(unet_model, valid_loader, device)
    #     print("---Unet---")
    #     print(f'epochs : {epoch}')
    #     print(
    #         f"train loss: {total_loss:.4f} | train dice score: {train_dice_score:.4f}")
    #     print(f"vaild dice score: {vaild_dice_score:.4f}")
    #     if vaild_dice_score > best_score:
    #         best_score = vaild_dice_score
    #         torch.save(unet_model, "saved_models/DL_Lab3_UNet_109550165_吳俊宏.pth")
    #         print('save checkpoint')
    resnet34_unet_model = resnet34_unet.Resnet34_unet(resnet34_unet.Block).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    lr = lr*0.1
    optimizer = optim.Adam(resnet34_unet_model.parameters(), lr=lr)
    best_score = 0 
    for epoch in range(1, epochs+1):
        total_loss = 0
        for i, data in enumerate(train_loader, 0):
            x_data = data['image'].to(device)
            y_label = data['mask'].to(device)
            y_pred = resnet34_unet_model(x_data)
            loss = loss_fn(y_pred, y_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_label.shape[0]
        total_loss /= len(train_loader)
        train_dice_score = evaluate.evaluate(resnet34_unet_model, train_loader, device)
        vaild_dice_score = evaluate.evaluate(resnet34_unet_model, valid_loader, device)
        print("---Resnet34_unet---")
        print(f'epochs : {epoch}')
        print(
            f"train loss: {total_loss:.4f} | train dice score: {train_dice_score:.4f}")
        print(f"vaild dice score: {vaild_dice_score:.4f}")
        if vaild_dice_score > best_score:
            best_score = vaild_dice_score
            torch.save(resnet34_unet_model, "saved_models/DL_Lab3_ResNet34_UNet_109550165_吳俊宏.pth")
            print('save checkpoint')               
def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int,
                        default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float,
                        default=1e-5, help='learning rate')
    parser.add_argument("--device", type=str,
                        default='cuda', help='model device')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
