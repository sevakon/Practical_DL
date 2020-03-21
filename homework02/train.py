import argparse
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.autograd import Variable
import PIL
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

from tiny_img import download_tinyImg200


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-batches_per_update", default=1, type=int)
    parser.add_argument("-image_size", default=64, type=int)
    parser.add_argument("-n_classes", default=200, type=int)
    parser.add_argument("-n_epochs", default=100, type=int)
    parser.add_argument("-lr", default=0.001, type=float)
    parser.add_argument("-firstlaunch", action="store_true")
    parser.add_argument("-checkpoint_head", action="store_true")
    parser.add_argument("-checkpoint_backbone", action="store_true")

    return parser.parse_args()


class MyVGGStyleModel(nn.Module):
    def __init__(self, checkpoint_head, checkpoint_backbone, n_classes):
        super(MyVGGStyleModel, self).__init__()
        self.checkpoint_head = checkpoint_head
        self.checkpoint_backbone = checkpoint_backbone

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.fcn = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        )

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of model's trainable parameters: {}".format(params))

    def forward(self, x):
        if self.checkpoint_backbone:
            x.requires_grad = True
            x = torch.utils.checkpoint.checkpoint(self.block1, x)
            x = torch.utils.checkpoint.checkpoint(self.block2, x)
            x = torch.utils.checkpoint.checkpoint(self.block3, x)
            x = torch.utils.checkpoint.checkpoint(self.block4, x)
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)

        x = nn.Flatten()(x)

        if self.checkpoint_head:
            out = torch.utils.checkpoint.checkpoint(self.fcn, x)
        else:
            out = self.fcn(x)

        return out


def create_val_folder(path, filename):
    """
    This method is responsible for separating validation images into separate sub folders
    """
    fp = open(filename, "r")
    data = fp.readlines()

    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


def get_train_and_val_loaders(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train', transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000])

    print('Train dataset size: {}, Val dataset size: {}'.format(len(train_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def get_test_loader(batch_size):
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/val/images', transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader


def train_one_epoch(model, device, loader, opt, criterion, batches_per_update):
    def compute_loss(X_batch, y_batch):
        X_batch = Variable(torch.FloatTensor(X_batch)).to(device)
        y_batch = Variable(torch.LongTensor(y_batch)).to(device)
        logits = model(X_batch)
        return F.cross_entropy(logits, y_batch).mean()

    model.train(True)
    epoch_train_losses = []

    batch_idx = 0
    for (X_batch, y_batch) in tqdm(loader):
        loss = compute_loss(X_batch, y_batch)
        loss.backward()

        if (batch_idx + 1) % batches_per_update == 0:
            opt.step()
            opt.zero_grad()

        epoch_train_losses.append(loss.cpu().data.numpy())
        batch_idx += 1

    epoch_train_loss = np.mean(epoch_train_losses)
    return epoch_train_loss


def eval_one_epoch(model, device, loader, criterion):
    model.train(False)
    epoch_val_losses, epoch_val_accs = [], []

    for X_batch, y_batch in tqdm(loader):

        logits = model(torch.FloatTensor(X_batch).to(device))
        loss = criterion(logits, torch.LongTensor(y_batch).to(device))

        y_pred = logits.max(1)[1].data

        epoch_val_losses.append(loss.cpu().item())
        epoch_val_accs.append(np.mean( (y_batch.cpu() == y_pred.cpu()).numpy() ))

    epoch_val_loss = np.mean(epoch_val_losses)
    epoch_val_acc = np.mean(epoch_val_accs)

    return epoch_val_loss, epoch_val_acc


def train(model, device, train_loader, val_loader, opt, criterion, batches_per_update, n_epochs, path):
    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        start_time = time.time()
        print('Epoch {} started... '.format(epoch + 1))

        train_loss = train_one_epoch(model, device, train_loader, opt, criterion, batches_per_update)
        print('  Train Loss: {}'.format(train_loss))
        train_losses.append(train_loss)

        val_loss, val_acc = eval_one_epoch(model, device, val_loader, criterion)
        print("  Validation loss: \t{:.4f}".format(val_loss))
        print("  Validation accuracy: \t\t\t{:.2f} %".format(val_acc * 100))
        val_losses.append(val_loss); val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('  Saving model weights....')
            torch.save(model.state_dict(), path)

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - start_time))
        print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")

    return train_losses, val_losses, val_accs


def evaluate(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    for X_batch, y_batch in tqdm(loader):

        logits = model(torch.FloatTensor(X_batch).to(device))
        y_predicted = logits.max(1)[1].cpu().numpy()

        y_true += list(y_batch.numpy())
        y_pred += list(y_predicted)

    test_accuracy = accuracy_score(y_true, y_pred)

    print("Final results:")
    print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))

    if test_accuracy * 100 > 40:
        print("Achievement unlocked: 110lvl Warlock!")
    elif test_accuracy * 100 > 35:
        print("Achievement unlocked: 80lvl Warlock!")
    elif test_accuracy * 100 > 30:
        print("Achievement unlocked: 70lvl Warlock!")
    elif test_accuracy * 100 > 25:
        print("Achievement unlocked: 60lvl Warlock!")
    else:
        print("We need more magic! Follow instructons below")


def main(args):

    DATA_PATH = '.'
    PATH = 'tiny-imagenet-200/val/images'
    ANNOTATIONS = 'tiny-imagenet-200/val/val_annotations.txt'

    if args.firstlaunch:
        download_tinyImg200(DATA_PATH)
        create_val_folder(PATH, ANNOTATIONS)

    train_loader, val_loader = get_train_and_val_loaders(args.batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyVGGStyleModel(args.checkpoint_head,
                            args.checkpoint_backbone,
                            args.n_classes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    opt.zero_grad()
    criterion = nn.CrossEntropyLoss()
    STATE_DICT_PATH = 'best_weights.pth'

    train_losses, val_losses, val_accs = train(model, device, train_loader, val_loader,
                                               opt, criterion, args.batches_per_update,
                                               args.n_epochs, STATE_DICT_PATH)

    print('Best train loss: {}'.format(min(train_losses)))
    print('Best val loss: {}'.format(min(val_losses)))
    print('Best validation accuracy: {:.2f}%'.format(max(val_accs) * 100))

    model.load_state_dict(torch.load(STATE_DICT_PATH))
    evaluate(model, device, get_test_loader(args.batch_size))


if __name__ == '__main__':
    main(get_args())
