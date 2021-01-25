# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
import os
from torch.utils.data import DataLoader


# def dataset_split(file):
#     delaney = pd.read_csv("delaney.csv")
#     test_set = delaney.sample(frac=0.1, random_state=0)
#     train_set = delaney.drop(test_set.index)
#     test_set.to_csv("delaney_test.csv", index=False)
#     train_set.to_csv("delaney_train.csv", index=False)


def train(model="nmr", epochs=80, device=th.device("cpu"), args=''):
    print("start")
    train_file = args.train_file
    valid_file = args.valid_file
    test_file = args.test_file
    save = args.save
    # train_dir = "./"
    # train_file = dataset+"_train.csv"
    if model == 'nmr':
        from Alchemy_dataset_nmr import TencentAlchemyDataset, batcher
        from sch_nmr import SchNetModel
    elif model == 'bde':
        from Alchemy_dataset_bde import TencentAlchemyDataset, batcher
        from sch_bde import SchNetModel
    elif model == 'literature':
        from Alchemy_dataset_litr import TencentAlchemyDataset, batcher
        from sch_litr import SchNetModel
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_dataset.mode = "Train"
    alchemy_dataset.transform = None
    alchemy_dataset.file_path = train_file
    alchemy_dataset._load()

    valid_dataset = TencentAlchemyDataset()
    # valid_dir = train_dir
    # valid_file = dataset+"_valid.csv"
    valid_dataset.mode = "Train"
    valid_dataset.transform = None
    valid_dataset.file_path = valid_file
    valid_dataset._load()

    test_dataset = TencentAlchemyDataset()
    # test_dir = train_dir
    # test_file = dataset+"_valid.csv"
    test_dataset.mode = "Train"
    test_dataset.transform = None
    test_dataset.file_path = test_file
    test_dataset._load()

    alchemy_loader = DataLoader(
        dataset=alchemy_dataset,
        batch_size=20,
        collate_fn=batcher(),
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=20,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=20,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )

    model = SchNetModel(norm=False, output_dim=1)

    print(model)
    # if model.name in ["MGCN", "SchNet"]:
    #     model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.to(device)
    # print("test_dataset.mean= %s" % (alchemy_dataset.mean))
    # print("test_dataset.std= %s" % (alchemy_dataset.std))

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = th.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.9, eta_min=0.0001, last_epoch=-1)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=10, threshold=0.0000001, threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08, verbose=False)

    log = open(save + '/train_log.txt', 'w')

    def print_res(label, res, op):
        size = len(res)
        for i in range(size):
            line = "%s,%s\n" % (label[i][0], res[i][0])
            op.writelines(line)

    def eval(model, epoch, data_loader, eval_mode, log):
        val_loss, val_mae = 0, 0
        for jdx, batch in enumerate(data_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)
            batch.graph0 = batch.graph0.to(device)
            batch.graph1 = batch.graph1.to(device)
            batch.graph2 = batch.graph2.to(device)
            batch.graph3 = batch.graph3.to(device)
            batch.graph4 = batch.graph4.to(device)
            batch.graph5 = batch.graph5.to(device)
            batch.graph6 = batch.graph6.to(device)
            batch.graph7 = batch.graph7.to(device)
            batch.feature = batch.feature.to(device)

            res = model(batch.graph, batch.graph0, batch.graph1, batch.graph2, batch.graph3, batch.graph4, batch.graph5, batch.graph6, batch.graph7,
                        batch.feature)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            val_mae += mae.detach().item()
            val_loss += loss.detach().item()
        val_mae /= jdx + 1
        val_loss /= jdx + 1
        print(
            f"Epoch {epoch} {eval_mode}_loss: {round(val_loss,4)}, {eval_mode}_mae: {round(val_mae,4)}")
        log.write(
            f"Epoch {epoch} {eval_mode}_loss: {round(val_loss,4)}, {eval_mode}_mae: {round(val_mae,4)}\n")

        if epoch % args.saveFreq == 0:
            # op_file = open('save/'+str(epoch)+'re')
            res_op = open(f'{save}/{eval_mode}_res_{epoch}.csv', 'w')
            th.save(model.state_dict(), save+'/model_'+str(epoch))
            for jdx, batch in enumerate(data_loader):
                batch.graph.to(device)
                batch.label = batch.label.to(device)
                batch.graph0 = batch.graph0.to(device)
                batch.graph1 = batch.graph1.to(device)
                batch.graph2 = batch.graph2.to(device)
                batch.graph3 = batch.graph3.to(device)
                batch.graph4 = batch.graph4.to(device)
                batch.graph5 = batch.graph5.to(device)
                batch.graph6 = batch.graph6.to(device)
                batch.graph7 = batch.graph7.to(device)
                batch.feature = batch.feature.to(device)

                res = model(batch.graph, batch.graph0, batch.graph1, batch.graph2, batch.graph3, batch.graph4, batch.graph5, batch.graph6, batch.graph7,
                            batch.feature)

                l = batch.label.cpu().detach().numpy()
                r = res.cpu().detach().numpy()
                print_res(l, r, res_op)
            res_op.close()

    for epoch in range(epochs):

        w_loss, w_mae = 0, 0
        model.train()

        for idx, batch in enumerate(alchemy_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)
            batch.graph0 = batch.graph0.to(device)
            batch.graph1 = batch.graph1.to(device)
            batch.graph2 = batch.graph2.to(device)
            batch.graph3 = batch.graph3.to(device)
            batch.graph4 = batch.graph4.to(device)
            batch.graph5 = batch.graph5.to(device)
            batch.graph6 = batch.graph6.to(device)
            batch.graph7 = batch.graph7.to(device)
            batch.feature = batch.feature.to(device)

            res = model(batch.graph, batch.graph0, batch.graph1, batch.graph2, batch.graph3, batch.graph4, batch.graph5, batch.graph6, batch.graph7,
                        batch.feature)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            mae.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()

        w_mae /= idx + 1
        w_loss /= idx + 1
        scheduler.step(w_mae)

        print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(
            epoch, w_loss, w_mae))
        log.write("Epoch {:2d}, loss: {:.7f}, mae: {:.7f} \n".format(
            epoch, w_loss, w_mae))
        model.eval()
        with th.no_grad():
            if epoch % args.saveFreq == 0:
                res_op = open(save+'/Train_res_'+str(epoch)+'.csv', 'w')
                for jdx, batch in enumerate(alchemy_loader):
                    batch.graph.to(device)
                    batch.label = batch.label.to(device)
                    batch.graph0 = batch.graph0.to(device)
                    batch.graph1 = batch.graph1.to(device)
                    batch.graph2 = batch.graph2.to(device)
                    batch.graph3 = batch.graph3.to(device)
                    batch.graph4 = batch.graph4.to(device)
                    batch.graph5 = batch.graph5.to(device)
                    batch.graph6 = batch.graph6.to(device)
                    batch.graph7 = batch.graph7.to(device)
                    batch.feature = batch.feature.to(device)

                    res = model(batch.graph, batch.graph0, batch.graph1, batch.graph2, batch.graph3, batch.graph4, batch.graph5, batch.graph6, batch.graph7,
                                batch.feature)
                    loss = loss_fn(res, batch.label)
                    mae = MAE_fn(res, batch.label)

                    l = batch.label.cpu().detach().numpy()
                    r = res.cpu().detach().numpy()
                    print_res(l, r, res_op)
                res_op.close()

        eval(model, epoch, valid_loader, 'valid', log)
        eval(model, epoch, test_loader, 'test', log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (nmr,bde)",
                        default="sch")
    parser.add_argument("--epochs", help="number of epochs", default=10000)
    parser.add_argument(
        "--train_file", help="dataset for training", default="")
    parser.add_argument(
        "--valid_file", help="dataset for validation", default="")
    parser.add_argument("--test_file", help="dataset for test", default="")
    parser.add_argument("--save", help="save option", default="")
    parser.add_argument("--device", help="device", default='cuda:0')
    parser.add_argument("--saveFreq", help="save frequency",
                        type=int, default=10)

    args = parser.parse_args()
    device = th.device(args.device if th.cuda.is_available() else 'cpu')
    assert args.model in ["bde", "nmr", "literature"]
    # dataset_split("delaney.csv")
    os.system(f'mkdir {args.save}')
    train(args.model, int(args.epochs), device, args)
