import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import pdb
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from models.resnet_cifar import ModifResNet18

CHECKPOINT_PATH = "saved_models/pseudo_cifar/teacher_best.pth.tar"
NO_ITER = 3


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train_and_evaluate(model, n_epochs, train_loader, val_loader, optimizer, loss_fn, device, iteration, save=False, scheduler=None):
    total = len(train_loader.dataset)
    print("Teacher network training, iteration: ", iteration)
    print("Total no. labelled images: ", total)
    best_acc = 0
    for epoch in range(1, n_epochs + 1):
        model.train()

        correct = 0
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            pred = torch.argmax(outputs.data, dim=1)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += pred.eq(labels.data.view_as(pred)).sum()

        print("Epoch: %d, Loss: %f, Acc: %f" % (epoch, loss, correct.item() / total))
        print("Current best accuracy: ", best_acc)

        new_acc = evaluate(model, val_loader, device)
        if new_acc > best_acc:
            print("New best accuracy: ", new_acc)
            best_acc = new_acc
            if save:
                save_checkpoint({
                    'iteration': iteration,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_acc': best_acc},
                    filename=CHECKPOINT_PATH)

        if scheduler is not None:
            scheduler.step()

    print("Final top accuracy: ", best_acc)


def evaluate(model, val_loader, device):
    total = len(val_loader.dataset)
    correct = 0
    model.eval()

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            pred = torch.argmax(outputs.data, dim=1)
            correct += pred.eq(labels.data.view_as(pred)).sum()

        print("Accuracy: %f" % (correct.item() / total))
        acc = correct.item() / total

    return acc


def generate_pseudolabels(model, data_loader, device):
    pseudo = []
    with torch.no_grad():
        for imgs, labs in tqdm(data_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = torch.argmax(outputs.data, dim=1)
            pseudo.append(pred)

    return pseudo


def pseudo_train(model, n_epochs, pseudo_labels, data_loader, optimizer, loss_fn, device, iteration, save=False):
    total = len(data_loader.dataset)
    print("Total no. unlabelled images: ", total)
    for epoch in range(1, n_epochs + 1):
        model.train()

        correct = 0
        true_correct = 0
        for (imgs, lab), labels in tqdm(zip(data_loader, pseudo_labels), total=len(pseudo_labels)):
            imgs, lab, labels = imgs.to(device), lab.to(device), labels.to(device)
            outputs = model(imgs)

            pred = torch.argmax(outputs.data, dim=1)
            loss = loss_fn(outputs, labels)
            correct += pred.eq(labels.data.view_as(pred)).sum()
            true_correct += pred.eq(lab.data.view_as(pred)).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: %d, Loss: %f, Acc: %f" % (epoch, loss, correct.item() / total))
        print("True acc: ", true_correct.item() / total)
        if save and epoch == n_epochs:
            torch.save(model.state_dict(), "saved_models/pseudo_cifar/trained_model_" + str(iteration) + ".pth")


def main():
    device = torch.device('cuda')
    epochs = 60
    batch_size = 128

    data_path = 'data/cifar/'
    cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                               ])
                               )
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                   ])
                                   )

    lab_data, unlab_data = split_dataset(cifar10, 0.98)

    train_loader = torch.utils.data.DataLoader(lab_data, batch_size=batch_size, shuffle=True)
    unlab_loader = torch.utils.data.DataLoader(unlab_data, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=100, shuffle=False)

    teacher_network = ModifResNet18(pretrained=False, n_classes=10, sem_seg=False)

    criterion = nn.CrossEntropyLoss()
    optimizerAdam = optim.Adam(teacher_network.parameters())
    optimizer = optim.SGD(teacher_network.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
    teacher_network.to(device)

    train_and_evaluate(teacher_network, 120, train_loader, val_loader, optimizer, criterion, device, 0, save=True, scheduler=scheduler)

    for i in range(1, NO_ITER + 1):
        checkpoint = torch.load(CHECKPOINT_PATH)
        print("Load model in iteration {0}, epoch {1}".format(checkpoint['iteration'], checkpoint['epoch']))
        print("Path loaded: ", CHECKPOINT_PATH)
        teacher_network.load_state_dict(checkpoint['state_dict'])
        teacher_network.eval()

        pseudo = generate_pseudolabels(teacher_network, unlab_loader, device)

        student_network = ModifResNet18(pretrained=False, n_classes=10, sem_seg=False)
        student_network.to(device)
        student_optim_adam = optim.Adam(student_network.parameters(), lr=0.0001)
        student_optim = optim.SGD(student_network.parameters(), lr=0.01,
                                  momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optim, T_max=200, eta_min=0)

        pseudo_train(student_network, epochs, pseudo, unlab_loader, student_optim, criterion, device, i, save=True)

        # fine-tuning on labelled data
        train_and_evaluate(student_network, 150, train_loader, val_loader, student_optim, criterion, device, i, save=True, scheduler=scheduler)



# val_split is the percentage of unlabeled data
def split_dataset(dataset, val_split=0.25):
    labeled_idx, unlabeled_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    labeled_dataset = Subset(dataset, labeled_idx)
    unlabeled_dataset = Subset(dataset, unlabeled_idx)
    return labeled_dataset, unlabeled_dataset


if __name__ == '__main__':
    main()
