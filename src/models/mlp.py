import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes, activation=nn.ReLU):
        super(MLP, self).__init__()

        assert isinstance(hidden_sizes, list) and len(hidden_sizes) > 0

        layer_list = [nn.Linear(in_dim, hidden_sizes[0], bias=False)]
        for i in range(1, len(hidden_sizes)):
            layer_list.extend([activation(), nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=False)])
        layer_list.extend([activation(), nn.Linear(hidden_sizes[-1], out_dim, bias=False)])
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))
