# -*- coding:utf-8 -*-

import dgl
import torch as th
import torch.nn as nn
import numpy as np


class DftModel(nn.Module):
    """
    a linear model for nmr chemical shift prediction
    """

    def __init__(self, poly_order=5):
        """
        Args:
        """
        super().__init__()
        self.poly_order = poly_order
        self.param = th.rand(poly_order)
        if th.cuda.is_available:
            self.param = self.param.cuda()
            self.param.requires_grad = True
        self.__parameters = dict(param=self.param)

    def make_features(self, x):
        # x = x.unsqueeze()
        return th.cat([x**i for i in range(0, self.poly_order)], 1)

    def poly_f(self, x):
        # print(f"x={x}")
        # print(f"para= {self.param}")
        y = x*self.param
        return y.sum(axis=1)

    def parameters(self):
        for name, value in self.__parameters.items():
            yield value

    def forward(self, dft):
        poly_dft = self.make_features(dft)

        pred = self.poly_f(poly_dft)

        # print('res after sum_nodes %s'%(res))
        return pred.unsqueeze(1)


if __name__ == "__main__":
    batch_size = 5
    dft_model = DftModel(4)
    x = th.FloatTensor([0, 1, 2])
    x = x.unsqueeze(1)
    print(x.cuda())
    x = dft_model(x.cuda())
    print(x)
    # x = x.mul(x)
    # print(x)
    # x = x.sum(dim=1)
    # print(x)
