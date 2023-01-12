import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from shufflenet import ShuffleNetG2
from dataset import SelfDataset
import torch.backends.cudnn as cudnn

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = ShuffleNetG2().to(device)
    model.load_state_dict(torch.load(".\\model\\train_model_epoch_44_acc_0.9860000014305115.pkl"))
    model.eval()
    test_dataset = SelfDataset(base_dir=".\\cancer_data\\test\\", transform=None)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False)
    total = test_dataset.__len__()
    correct_cnt = 0
    pbar = tqdm(test_loader, mininterval=0.3)
    cudnn.benchmark = True
    torch.cuda.empty_cache()
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        pred = torch.argmax(output, 1)
        correct_cnt += (pred == torch.argmax(target, 1)).sum().float()

    print(f"accuracy: {float(correct_cnt / total)}")
    print(f"correct: {int(correct_cnt)}, total: {total}")
