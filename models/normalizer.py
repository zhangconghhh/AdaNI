import torch, pdb
import torch.nn as nn

class Normalize_layer(nn.Module):

    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)

    def forward(self, x, label, mixBN):
        # x = x.cuda()
        return x.sub(self.mean).div(self.std), label, mixBN
