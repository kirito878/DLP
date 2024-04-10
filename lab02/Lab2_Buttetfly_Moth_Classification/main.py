import dataloader
import ResNet50
import VGG19
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import argparse


def show_result(epochs, vgg19_valid_acc, vgg19_train_acc,resnet50_valid_accs,resnet50_train_accs):
    plt.plot(epochs, vgg19_train_acc, linestyle='-', label='VGG19_train_acc')
    plt.plot(epochs, vgg19_valid_acc, linestyle='-', label='VGG19_valid_acc')
    plt.plot(epochs, resnet50_train_accs, linestyle='-', label='Resnet50_train_accs')
    plt.plot(epochs, resnet50_valid_accs, linestyle='-', label='Resnet50_valid_accs')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig('plot/acc_curve.png')


def accuracy(predictions, ground_truth):
    if len(predictions) != len(ground_truth):
        raise ValueError("y_true and y_pred must have the same length")
    correct = sum(1 for true, pred in zip(
        ground_truth, predictions) if true == pred)
    length = len(ground_truth)
    accuracy = correct / length
    # print(correct)
    return accuracy


def evaluate(model, valid_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * y.shape[0]
            _, predict = torch.max(y_pred, 1)
            predictions.extend(predict.cpu().numpy())
            ground_truth.extend(y.cpu().numpy())
        total_loss /= len(predictions)
        valid_acc = accuracy(predictions, ground_truth)
    return total_loss, valid_acc


def evaluate_noloss(model, data_loader, device):
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            _, predict = torch.max(y_pred, 1)
            predictions.extend(predict.cpu().numpy())
            ground_truth.extend(y.cpu().numpy())
        acc = accuracy(predictions, ground_truth)
        # print(len(predictions))
    return acc


def test(train_loader, valid_loader, test_loader, device):
    test_vgg19 = VGG19.VGG19_nn().to(device)
    vgg19_weight_path = "model_weight/vgg19.pth"
    test_vgg19.load_state_dict(torch.load(vgg19_weight_path))
    train_acc = evaluate_noloss(test_vgg19, train_loader, device)
    valid_acc = evaluate_noloss(test_vgg19, valid_loader, device)
    test_acc = evaluate_noloss(test_vgg19, test_loader, device)
    name = "VGG19"
    print("---VGG19---")
    print(f'{name :<10s}|train accuracy: {train_acc*100:>10.3f}%|valid accuracy: {valid_acc*100:>10.3f}%|test accuracy: {test_acc*100:>10.3f}%')
    test_resnet50 = ResNet50.Resnet50_net(ResNet50.Bottleneck).to(device)
    resent_weight_path = 'model_weight/resnet50.pth'
    test_resnet50.load_state_dict(torch.load(resent_weight_path))
    train_acc = evaluate_noloss(test_resnet50, train_loader, device)
    valid_acc = evaluate_noloss(test_resnet50, valid_loader, device)
    test_acc = evaluate_noloss(test_resnet50, test_loader, device) 
    name = "Resnet50"
    print("---Resnet50---")
    print(f'{name :<10s}|train accuracy: {train_acc*100:>10.3f}%|valid accuracy: {valid_acc*100:>10.3f}%|test accuracy: {test_acc*100:>10.3f}%')



def train(model, loss_fn, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    predictions = []
    ground_truth = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item() * y.shape[0]

        with torch.no_grad():
            _, predict = torch.max(y_pred, 1)
            predictions.extend(predict.cpu().numpy())
            ground_truth.extend(y.cpu().numpy())
    total_loss /= len(predictions)
    train_accuracy_score = accuracy(predictions, ground_truth)
    return total_loss, train_accuracy_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", default="train")
    args = parser.parse_args()
    mode = args.mode
    if mode == "train":
        inference = False
    else:
        inference = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = dataloader.BufferflyMothLoader(
        "/home/wujh1123/DLP/lab02/Lab2_Buttetfly_Moth_Classification/dataset", mode='train', inference=inference)
    valid_data = dataloader.BufferflyMothLoader(
        "/home/wujh1123/DLP/lab02/Lab2_Buttetfly_Moth_Classification/dataset", mode='valid', inference=True)
    test_data = dataloader.BufferflyMothLoader(
        root="/home/wujh1123/DLP/lab02/Lab2_Buttetfly_Moth_Classification/dataset", mode='test', inference=True)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    if mode == 'train':
        print("Trian mode")
        vgg19 = VGG19.VGG19_nn().to(device)
        # hyper
        epochs_list = []
        vgg19_train_accs = []
        vgg19_valid_accs = []
        lr = 1e-3
        epochs = 60
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(vgg19.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.3)
        print(f'learn rate: {lr}')
        best_accuracy = 0
        for epoch in range(1, epochs+1):
            train_loss, train_accuracy = train(
                vgg19, loss_fn, optimizer, train_loader, device)
            valid_loss, valid_accuracy = evaluate(
                vgg19, valid_loader, loss_fn, device)
            epochs_list.append(epoch)
            vgg19_train_accs.append(train_accuracy*100)
            vgg19_valid_accs.append(valid_accuracy*100)
            print("-----VGG19-----")
            print(f'epochs : {epoch} ')
            print(
                f"train accuracy: {train_accuracy*100:.4f}%  train loss: {train_loss:.4f}")
            print(
                f"valid accuracy: {valid_accuracy*100:.4f}%  valid loss: {valid_loss:.4f}")
            scheduler.step()
            print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(vgg19.state_dict(), "model_weight/vgg19.pth")
                print('save checkpoint')
        resnet50 = ResNet50.Resnet50_net(ResNet50.Bottleneck).to(device)
        resnet50_train_accs = []
        resnet50_valid_accs = []
        lr = 1e-3
        epochs = 60
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(resnet50.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.3)
        print(f'learn rate: {lr}')
        best_accuracy = 0
        for epoch in range(1, epochs+1):
            train_loss, train_accuracy = train(
                resnet50, loss_fn, optimizer, train_loader, device=device)
            valid_loss, valid_accuracy = evaluate(resnet50,valid_loader,loss_fn,device)
            resnet50_train_accs.append(train_accuracy*100)
            resnet50_valid_accs.append(valid_accuracy*100)
            print("---Resnet50---")
            print(f'epochs : {epoch}')
            print(
                f"train accuracy: {train_accuracy*100:.4f}%  train loss: {train_loss:.4f}")
            print(
                f"valid accuracy: {valid_accuracy*100:.4f}%  valid loss: {valid_loss:.4f}")
            scheduler.step()
            print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(resnet50.state_dict(), "model_weight/resnet50.pth")
                print('save checkpoint')   
        show_result(epochs_list, vgg19_valid_accs, vgg19_train_accs,resnet50_valid_accs,resnet50_train_accs)
    elif mode == "test":
        test(train_loader, valid_loader, test_loader, device)
    else:
        print("Error mode")
