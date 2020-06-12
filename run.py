import torch as tc
import torch.nn as nn
import torch.utils.data as Data
import numpy as npy
import pandas as pds
from pathlib import Path
from configparser import ConfigParser
import argparse
import os
from utils.metrics import cluster_acc, cluster_nmi, cluster_ari, cluster_ri_
from layers.encoder import CnnEncoder, LstmEncoder, CnnLstmEncoder, CatEncoder


class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()

    def forward(self, x):
        x = x / tc.sqrt(tc.sum(x**2, 1, keepdim=True))
        union = tc.matmul(x, x.T)
        union[union.ge(1.)] = 1.
        return union


def metrics_loss(x, y):
    return tc.nn.BCELoss(reduction='mean')(x, y)


def DataInit(NAME, flag='TEST'):
    p = Path(f'Univariate_arff/{NAME}/{NAME}_{flag}.txt')
    f = npy.loadtxt(p)
    x_train = f[:, 1:]
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_train = npy.expand_dims(x_train, 2)
    y_train = f[:, 0]
    y_train[y_train == -1] = 0
    k_cluster = int(y_train.max() - y_train.min() + 1)
    print(f'name:{NAME}, shape:{x_train.shape}, class:{k_cluster}')
    return tc.tensor(x_train, dtype=tc.float32), tc.tensor(y_train), k_cluster


if __name__ == "__main__":
    # ----------------Params-------------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='CBF',type=str, help='UCR univariate dataset')
    parser.add_argument('--batch', default='128',type=int, help='batch size')
    parser.add_argument('--epoch', default='200',type=int, help='number of iteration')
    parser.add_argument('--lr', default='0.001',type=float, help='learning rate')
    parser.add_argument('--up', default='1.0',type=float, help='upper bound of threshold for DLG')
    parser.add_argument('--dw', default='0.8',type=float, help='lower bound of threshold for DLG')
    parser.add_argument('--period', default='10',type=int, help='period of updating threshold for DLG')
    parser.add_argument('--type', default='CR',type=str, help='type of encoder; C for CNN, R for RNN, CR for CNN+RNN')
    args = parser.parse_args()
    print("configs: ", args)
    NAME = args.dataset
    BATCH_SIZE = args.batch
    EPOCH = args.epoch
    LR = args.lr
    UP = args.up
    DW = args.dw
    PERIOD = args.period
    TYPE = args.type
    DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")
    Path(f'results/{NAME}/').mkdir(exist_ok=True)

    # ---------------Data----------------------------------
    x_train, y_train, k_cluster = DataInit(NAME, 'TEST')
    x_train = x_train.to(DEVICE)
    DW = max(DW, (k_cluster-1)/k_cluster)
    L = (UP - DW) / EPOCH * PERIOD
    x_size = x_train.shape
    BATCH_SIZE = min(BATCH_SIZE, x_size[0])

    # ---------------Model----------------------------
    data_set = Data.TensorDataset(x_train, y_train)
    loader = Data.DataLoader(
        dataset=data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    if TYPE == 'CR':
        enc = CnnLstmEncoder(1, k_cluster).to(DEVICE)
    elif TYPE =='CR+':
        enc = CnnLstmEncoder(1, k_cluster, kernel=[5, 15, 25], hidden=[32, 32, 32, 128]).to(DEVICE)
    elif TYPE == 'C':
        enc = CnnEncoder(1, k_cluster).to(DEVICE)
    elif TYPE == 'R':
        enc = LstmEncoder(1, k_cluster).to(DEVICE)

    met = Metric().to(DEVICE)
    optimizer = tc.optim.Adam(enc.parameters(), lr=LR)

    # -------------------Training---------------------------
    print("***Begin Training***")
    results = {'acc': [], 'nmi': [], 'ari': [], 'ri': []}
    for epoch in range(EPOCH):
        
        print(f"Epoch {epoch+1}: {epoch+1}/{EPOCH}; <theta: {UP:.3f}>; <lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}>")
        loss_sum = 0
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(DEVICE)
            x_embedded = enc(x_batch)
            x_union = met(x_embedded)
            y_union = x_union.ge(UP).float()
            loss = metrics_loss(x_union, y_union)
            loss_sum += loss
            print(f'\rBatch: [{BATCH_SIZE*(step+1)}/{x_size[0]}]; loss: {float(loss_sum / (step + 1)):.3f}', end="")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        x_embedded = enc(x_train)
        ans = tc.argmax(x_embedded, -1)
        y_true = y_train.numpy()
        y_pred = ans.cpu().numpy()
        print()
        results['acc'].append(cluster_acc(y_true, y_pred))
        results['nmi'].append(cluster_nmi(y_true, y_pred))
        results['ari'].append(cluster_ari(y_true, y_pred))
        results['ri'].append(cluster_ri_(y_true, y_pred))
        if (epoch + 1) % PERIOD == 0:
            UP -= L
        if (epoch+1) % 10 == 0:
            print(f"acc: {results['acc'][-1]:.4f}; nmi: {results['nmi'][-1]:.4f}; ari: {results['ari'][-1]:.4f}; ri: {results['ri'][-1]:.4f}", end="\n\n")

    # -------------Test---------------------------------
    x_test, y_test, k_cluster = DataInit(NAME, 'TEST')
    x_test = x_test.to(DEVICE)
    x_embedded = enc(x_test)
    ans = tc.argmax(x_embedded, -1)
    y_pred = ans.cpu().numpy()
    y_true = y_test.numpy()

    # ------------Print----------------------------------
    print('***results***')
    print(f"{NAME}, acc: {results['acc'][-1]:.4f}, nmi: {results['nmi'][-1]:.4f}, ari: {results['ari'][-1]:.4f}, ri: {results['ri'][-1]:.4f}")

    #-------------Save----------------------------------
    # results = pds.DataFrame(results)
    # results.to_csv(f'results/{NAME}/metrics.csv', index=False)
