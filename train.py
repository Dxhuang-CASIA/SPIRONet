import os
import random

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import seg_dataset, RandomGenerator
import torch.backends.cudnn as cudnn

from model.network import Segmodel


class BCELoss(nn.Module):
    def __init__(self, reduction = "mean", pos_weight = 1.0):
        pos_weight = torch.tensor(pos_weight).cuda(0)
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction = reduction, pos_weight = pos_weight
        )

    def forward(self, predictions, targets):
        return self.bce_loss(predictions, targets)


def seed_torch(seed):
    # 随机种子
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train(
        seed,
        base_path,
        dataset_name,
        image_size,
        batch_size,
        lr,
        weight_decay,
        epochs,
        ckpt
):
    seed_torch(seed)

    train_dataset = seg_dataset(base_path = base_path, name = dataset_name, mode = 'train', transform = RandomGenerator(), image_size = image_size)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16, pin_memory = False)

    model = Segmodel(in_channels = 1, num_classes = 1, img_size = image_size, device = 'cuda:0').cuda(0)

    loss_ce = BCELoss()

    optimizer = torch.optim.SGD(model.parameters(),  lr = lr, momentum = 0.9, weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / epochs) ** 0.9)

    loss_list = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_dataloader), total = len(train_dataloader), ncols = 160)
        for step, batch in loop:
            img = batch['img'].cuda(0)
            label = batch['label'].cuda(0).unsqueeze(1)

            pred = model(img)
            loss = loss_ce(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            loop.set_description(
                'TRAIN ({}) | Loss: {:.4f} |'.format(
                    epoch + 1, np.mean(loss_list)))

        lr_scheduler.step()
        loss_list = []

        if (epoch + 1) % epochs == 0:
            path = f'seed{seed}.pth' % (epoch + 1)
            torch.save(model.state_dict(), os.path.join(ckpt, path))


if __name__ == '__main__':
    seed_list = [3407, 42, 924]
    base_path = 'path of your datasets'
    dataset_name = 'XCAD' # ['DCA1', 'XCAD', 'CAXF', 'CADSA']
    image_size = 512 # 'DCA1': 300, else: 512
    batch_size = 4
    lr = 0.06 # 'DCA1': 600, 'XCAD': 0.06, 'CAXF': 0.08, 'CADSA': 0.03
    weight_decay = 1e-4
    epochs = 700 # 'DCA1': 0.045, 'XCAD': 700, 'CAXF': 800, 'CADSA': 200

    print(dataset_name)

    for seed in seed_list:
        print('*' * 20)
        print(seed)
        ckpt = 'path to save your checkpoint'
        torch.autograd.set_detect_anomaly(True)
        train(seed, base_path, dataset_name, image_size, batch_size, lr, weight_decay, epochs, ckpt)
        torch.autograd.set_detect_anomaly(False)