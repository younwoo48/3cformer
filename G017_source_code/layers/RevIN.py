# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, channel_groups=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    @staticmethod
    def avg_stds(std_dev, channel_groups):
        result = torch.zeros(std_dev.size(0), 1, len(channel_groups))

        for batch_index in range(std_dev.size(0)):
            for group_index, group in enumerate(channel_groups):
                # Select the channels for the current group
                selected_channels = std_dev[batch_index, 0, group]

                # Sum the selected channels
                group_sum = selected_channels.sum() / len(group)

                # Calculate the divisor
                divisor = 1.2 + (0.33 * len(group))

                # Compute the result for the current group and assign it
                result[batch_index, 0, group_index] = group_sum / divisor
        return result
    
    
    @staticmethod
    def avg_mean(mean, channel_groups):
        result = torch.zeros(mean.size(0), 1, len(channel_groups))

        for batch_index in range(mean.size(0)):
            for group_index, group in enumerate(channel_groups):
                # Select the channels for the current group
                selected_channels = mean[batch_index, 0, group]

                # Sum the selected channels
                group_sum = selected_channels.sum() / len(group)

                # Compute the result for the current group and assign it
                result[batch_index, 0, group_index] = group_sum 
        return result
    
        
    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        # self.stdev = RevIN.avg_stds(self.stdev, channel_groups)
        x = x * self.stdev
        if self.subtract_last:
            # print("sub")
            # print(x.shape)
            # print(self.last.shape)
            x = x + self.last
        else:
            # print("mean")
            # print(x.shape)
            # print(self.mean.shape)
            # print(self.mean)
            # self.mean = RevIN.avg_mean(self.mean, channel_groups)
            x = x + self.mean
        return x
