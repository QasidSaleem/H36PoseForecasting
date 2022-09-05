import torch
from torch import nn
import torch.nn.functional as F

def get_dropout(drop_p):
    """ Getting a dropout layer """
    if(drop_p):
        drop = nn.Dropout(p=drop_p)
    else:
        drop = nn.Identity()
    return drop


def get_act(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)

    return activation()

class LinearBlock(nn.Module):
    """Applies Linear Transforms to the Input"""
    def __init__(self, input_dim, sizes=[128], act="ReLU", dropout=0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.sizes = sizes
        self.act = act
        self.dropout = dropout
        self.linear_block = self._create_block()
    
    def _create_block(self):
        layers = []
        in_size = self.input_dim

        for size in self.sizes:
            layers.append(nn.Linear(in_features=in_size, out_features=size))
            if self.act:
                layers.append(get_act(self.act))
            layers.append(get_dropout(self.dropout))
            in_size = size
        
        linear_block = nn.Sequential(*layers)
        return linear_block
    
    def forward(self, x):
        return self.linear_block(x)

def init_state(cell_type, rnn_dim, b_size, x):
    if cell_type == "LSTM":
        # type_as lightning thing
        # refer to https://pytorch-lightning.readthedocs.io/en/latest/accelerators/accelerator_prepare.html
        h = torch.zeros(b_size, rnn_dim).type_as(x)
        c = torch.zeros(b_size, rnn_dim).type_as(x)
        state = (h, c)
    else:
        state = torch.zeros(b_size, rnn_dim).type_as(x)
    
    return state