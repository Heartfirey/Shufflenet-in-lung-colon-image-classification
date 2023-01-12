import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from shufflenet import ShuffleNetG2
from dataset import SelfDataset


def work(use_cuda=True, random_seed=114514, epochs=30, learning_rate=0.0001):
    global best_acc
    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(random_seed)
    else:
        torch.manual_seed(random_seed)

    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_data = SelfDataset('.\\cancer_data\\train\\', transform=None)
    validate_data = SelfDataset('.\\cancer_data\\val\\', transform=None)
    train_loader = DataLoader(dataset=train_data, batch_size=50, shuffle=True, **kwargs)
    validate_loader = DataLoader(dataset=validate_data, batch_size=40, shuffle=True, **kwargs)
    model = ShuffleNetG2().to(device)

    if use_cuda:
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device.count()))
        cudnn.benchmark = True
        torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    writer = SummaryWriter(log_dir=".\\log")

    for epoch in range(1, epochs + 1):
        # Train Part
        model.train()
        average_loss = []
        pbar = tqdm(train_loader, desc=f'Train Epoch{epoch}/{epochs}')
        batch = 1
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            average_loss.append(loss.item())
            optimizer.step()
            pbar.set_description(f'Train Epoch:{epoch}/{epochs} train_loss:{round(np.mean(average_loss), 2)}')
            batch += 1
        avg_loss_train = np.array(average_loss).sum() / len(average_loss)
        writer.add_scalar(tag="loss/train", scalar_value=avg_loss_train, global_step=epoch)
        scheduler.step()

        # Validate Part
        model.eval()
        test_loss = 0
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        average_loss = []
        pbar = tqdm(validate_loader, desc=f'Test Epoch{epoch}/{epochs}', mininterval=0.3)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = model(data)
            now_loss = criterion(output, target).item()
            average_loss.append(now_loss)
            test_loss += now_loss
            pred = torch.argmax(output, 1)
            correct += (pred == torch.argmax(target, 1)).sum().float()
            total += len(target)
            predict_acc = correct / total
            pbar.set_description(f'Test Epoch:{epoch}/{epochs} test_loss:{round(np.mean(average_loss), 2)} predict_acc:{predict_acc}')
        avg_loss_validate = np.array(average_loss).sum() / len(average_loss)
        writer.add_scalar(tag="loss/validate", scalar_value=avg_loss_validate, global_step=epoch)
        writer.add_scalar(tag="accuracy/validate", scalar_value=(correct / total), global_step=epoch)
        if (correct / total) > 0.97:
            torch.save(model.state_dict(), f".\\model\\train_model_epoch_{epoch}_acc_{predict_acc}.pkl")

    samplex = Variable(torch.randn(1, 3, 32, 32)).cuda()
    torch.onnx.export(model, samplex, "onnx_final_model.onnx",  export_params=True, verbose=True)
    torch.save(model.state_dict(), ".\\model\\final_model.pkl")

if __name__ == "__main__":
    work(use_cuda=True, epochs=60)