import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from model.network import Segmodel

from dataset import seg_dataset
from torch.utils.data import DataLoader
from metric import get_metrics


def test(
        base_path,
        name,
        image_size,
        ckpt,
        seed,
        save_vis
):
    test_dataset = seg_dataset(base_path = base_path, name = name, mode = 'test', transform = None, image_size = image_size)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)


    model = Segmodel(in_channels = 1, num_classes = 1, img_size = image_size, device = 'cuda:0').cuda(0)

    path = f'seed{seed}.pth'
    print(path)
    model.load_state_dict(torch.load(os.path.join(ckpt, path)))

    f1_list = []
    sen_list = []
    iou_list = []
    mcc_list = []

    loop = tqdm(enumerate(test_dataloader), total = len(test_dataloader), ncols = 160)
    model.eval()
    for step, batch in loop:
        img = batch['img'].cuda(0)
        label = batch['label'].cuda(0)

        pred = model(img)
        pred = pred[0, 0]
        label = label[0]

        metric = get_metrics(pred, label)

        f1_list.append(metric["F1"])
        sen_list.append(metric["Sen"])
        iou_list.append(metric["IOU"])
        mcc_list.append(metric["MCC"])

        if save_vis:
            pred = torch.sigmoid(pred).cpu().detach().numpy()
            pred = np.where(pred >= 0.5, 1, 0) * 255
            pred = pred.astype(np.uint8)
            pred = Image.fromarray(pred, 'L')
            pred.save("path of your visualization results")

    print("Sen:%.4f" % np.mean(sen_list))
    print("F1:%.4f" % np.mean(f1_list))
    print("IOU:%.4f" % np.mean(iou_list))
    print("MCC:%.4f" % np.mean(mcc_list))

if __name__ == '__main__':

    seed_list = [3407, 42, 924]
    base_path = 'path of your datasets'
    dataset_name = 'XCAD'  # ['DCA1', 'XCAD', 'CAXF', 'CADSA']
    image_size = 512  # 'DCA1': 300, else: 512
    save_vis = False

    for seed in seed_list:
        print('*' * 20)
        print(seed)
        ckpt = f'../../../../../devdata1/hdxdata/vessel_datasets/my_work03/ckpt/ours/{dataset_name}/model4'
        test(base_path, dataset_name, image_size, ckpt, seed, save_vis)