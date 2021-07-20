from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict

from data.camvid import CamVidDataset, camvid_colors
from models.resnet_single_scale import *
from models.semseg import *
from data.transforms import *


CHECKPOINT_PATH = "saved_models/pseudo_cam/teacher_best.pth.tar"


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


camvid_colors = OrderedDict([
        (0, (128, 0, 0)),     # 0 Building
        (1, (128, 128, 0)),   # 1 Tree
        (2, (128, 128, 128)), # 2 Sky
        (3, (64, 0, 128)),    # 3 Car
        (4, (192, 128, 128)), # 4 SignSymbol
        (5, (128, 64, 128)),  # 5 Road
        (6, (64, 64, 0)),     # 6 Pedestrian
        (7, (64, 64, 128)),   # 7 Fence
        (8, (192, 192, 128)), # 8 Column_Pole
        (9, (0, 0, 192)),     # 9 Sidewalk
        (10, (0, 128, 192)),  # 10 Bicyclist
        (11, (0, 0, 0))])     # 11 Void


def collect_confusion_matrix(y, yt, conf_mat):
    size = y.size
    num_classes = conf_mat.shape[0]
    for i in range(size):
        pred = y[i]
        target = yt[i]
        if 0 <= target < num_classes:
            conf_mat[pred, target] += 1


def compute_errors(conf_mat, verbose=True):
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(1)
    TPFN = conf_mat.sum(0)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    per_class_iou = []
    if verbose:
        print('Errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = camvid_colors[i]
        per_class_iou += [(class_name, class_iou[i])]
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print('mean class precision -> TP / (TP+FP) = %.2f %%' % avg_class_precision)
        print('pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size, per_class_iou


def train(model, epochs, iteration, train_loader, val_loader, optimizer, scheduler, eval_each, eval_train=False, save=False):
    best_iou = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()

        print('Train| Epoch {}| Loss: {:.6f}'.format(
            epoch, loss.item(),
        ))

        if epoch % eval_each == 0:
            if eval_train:
                _, _ = evaluate(model, train_loader)
            new_iou, per_class_iou = evaluate(model, val_loader)
            if new_iou > best_iou:
                best_iou = new_iou
                best_epoch = epoch
                if save:
                    print("Saving model state...")
                    print("Iteration: ", iteration)
                    print("Epoch: ", epoch)
                    print("Best iou: ", best_iou)
                    print("Checkpoint path: ", CHECKPOINT_PATH)
                    save_checkpoint({
                        'iteration': iteration,
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_iou': best_iou},
                        filename=CHECKPOINT_PATH)

        scheduler.step()

    print("Best mIoU: ", best_iou)
    print("Best epoch: ", best_epoch)


def pseudo_train(model, epochs, data_loader, optimizer):
    for epoch in range(1, epochs + 1):
        model.train()

        for batch in tqdm(data_loader):
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()

        print('Train| Epoch {}| Loss: {:.6f}'.format(
            epoch, loss.item(),
        ))


def evaluate(model, data_loader, show=0):
    model.eval()
    conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
    total_shown = 0
    for batch in tqdm(data_loader):
        batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
        logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
        pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)

        if total_shown < show:
            for i in range(len(batch['original_labels'])):
                plt.figure()
                plt.title("CamVid test set: Prediction {0}".format(batch['name']))
                plt.imshow(pred[i])
                plt.show()

                plt.figure()
                plt.title("CamVid test set: Ground truth")
                plt.imshow(batch['original_labels'][i])
                plt.show()

                total_shown += 1

        collect_confusion_matrix(pred.flatten(), batch['original_labels'].flatten(), conf_mat)
    pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_mat)
    return iou_acc, per_class_iou


def generate_pseudo(model, data_loader, iteration, create_pseudo=False, show_each=100):
    model.eval()
    print("Generating pseudolabels...")
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader)):
            logits, _ = model.do_forward(batch, (720, 960))
            pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy()
            if create_pseudo:
                for name, img in zip(batch['name'], pred):
                    cv2.imwrite("data/camvid/unlabeled/train_pseudoIt{0}/{1}.png".format(iteration, name), img)

            if step % show_each == 0:
                plt.figure()
                plt.title("CamVid Pseudolabels, iter: {0}, img: {1}".format(iteration, batch['name'][0]))
                plt.imshow(pred[0])
                plt.show()


def main():
    NO_ITER = 3
    sup_epochs = 200
    semsup_epochs = 100
    device = torch.device('cuda')
    batch_size = 12
    random_crop_size = 448
    target_size_crops = (random_crop_size, random_crop_size)
    target_size_crops_feats = (random_crop_size // 4, random_crop_size // 4)

    scale = 1
    # mean = [73.15, 82.90, 72.3]
    # mean = [105.1305, 108.5845, 110.4605]
    mean = [111.376, 63.110, 83.670]

    std = [41.608, 54.237, 68.889]

    eval_each = 10

    mean_rgb = tuple(np.uint8(scale * np.array(mean)))
    size_cam = (720, 960)
    size_cam_feat = (720 // 4, 960 // 4)

    trans_train = Compose(
        [Open(),
         RandomFlip(),
         RandomSquareCropAndScale(random_crop_size, ignore_id=11, mean=mean_rgb),
         SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
         Tensor(),
         ]
    )

    trans_val = Compose(
        [Open(),
         SetTargetSize(target_size=size_cam, target_size_feats=size_cam_feat),
         Tensor(),
         ]
    )

    dataset_cam_train = CamVidDataset("./data/camvid", transforms=trans_train, subset='train')
    train_loader_cam = DataLoader(dataset=dataset_cam_train, batch_size=batch_size, shuffle=True)

    dataset_cam_val = CamVidDataset("./data/camvid", trans_val, subset='val')
    valid_loader_cam = DataLoader(dataset=dataset_cam_val, batch_size=1, shuffle=True)

    dataset_cam_test = CamVidDataset("./data/camvid", trans_val, subset='test')
    test_loader_cam = DataLoader(dataset=dataset_cam_test, batch_size=1, shuffle=True)

    # unlabeled dataset for generating pseudolabels
    pseudo_gen_dataset = CamVidDataset("./data/camvid/unlabeled", transforms=trans_val, pseudo=True, subset='train')
    pseudo_gen_loader = DataLoader(dataset=pseudo_gen_dataset, batch_size=1, shuffle=False)

    # model
    resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
    teacher_network = SemsegModel(resnet, num_classes=11)
    teacher_network.to(device)
    teacher_network.criterion = SemsegCrossEntropy(num_classes=11, ignore_id=11)
    
    # model was firstly trained with supervised learning for 400 epochs, with optim and scheduler as follows:

    # optimizer = optim.Adam(teacher_network.parameters(), lr=4e-3, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)
    # train(teacher_network, 400, 0, train_loader_cam, valid_loader_cam, optimizer, scheduler, eval_each, save=True)
    # evaluate(teacher_network, test_loader_cam, show=5)

    for i in range(1, NO_ITER + 1):
        checkpoint = torch.load(CHECKPOINT_PATH)
        print("Load model in iteration {0}, epoch {1}".format(checkpoint['iteration'], checkpoint['epoch']))
        print("Path loaded: ", CHECKPOINT_PATH)
        teacher_network.load_state_dict(checkpoint['state_dict'])
        teacher_network.eval()

        generate_pseudo(teacher_network, pseudo_gen_loader, i, create_pseudo=True)

        pseudolabels_dataset = CamVidDataset("./data/camvid/unlabeled", transforms=trans_train, subset='train', iteration=i)
        pseudolabels_loader = DataLoader(dataset=pseudolabels_dataset, batch_size=batch_size, shuffle=True)

        resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=scale)
        student_network = SemsegModel(resnet, num_classes=11)
        student_network.criterion = SemsegCrossEntropy(num_classes=11, ignore_id=11)
        student_network.to(device)
        student_optim = optim.Adam(student_network.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optim, T_max=300, eta_min=1e-6)

        pseudo_train(student_network, semsup_epochs, pseudolabels_loader, student_optim)

        # fine-tuning on labeled data
        train(student_network, sup_epochs, i, train_loader_cam, valid_loader_cam, student_optim, scheduler, eval_each, save=True)
        evaluate(student_network, test_loader_cam, show=5)



if __name__ == '__main__':
    main()
