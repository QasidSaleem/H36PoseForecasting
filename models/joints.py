import torch
from torch import nn
import torch.nn.functional as F

from .utils import get_dropout, get_act
from .utils import LinearBlock

class StateSpace2dJoints(nn.Module):
    """"""
    def __init__(
        self,
        input_dim=34,
        encoder_layers=[128],
        cell_type="LSTM",
        rnn_dim=64,
        residual_step=True,
        n_seeds=10,
        decoder_fixed_input=True,
        **kwargs
    ):
        super().__init__()
        # Linear encoder
        self.encoder = LinearBlock(input_dim, encoder_layers, **kwargs)
        rnn_cell = nn.LSTMCell if cell_type=="LSTM" else nn.GRUCell

        if not decoder_fixed_input:
            assert encoder_layers[-1] == rnn_dim, "To have the same input as"\
            " output, rnn input and output should have the same dimesions"\
            "either set decoder_fixed_input=True or set the last layer of"\
            " encoder and rnn_dim to be the same"
        self.rnn_cell = rnn_cell(encoder_layers[-1], rnn_dim)
        # if add_residual:
        #     self.rnn_cell = ResidualWrapper(self.rnn_cell)
        self.decoder = LinearBlock(rnn_dim, sizes=[input_dim], act=None)
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.residual_step = residual_step
        self.n_seeds = n_seeds
        self.decoder_fixed_input = decoder_fixed_input
    

    def forward(self, x):
        b_size, n_seqs, n_joints = x.shape
        # initial input of rnn
        state = self.init_state(b_size=b_size, x=x)
        embeddings = self.encoder(x[:, 0:self.n_seeds, :])

        # Encoding Steps
        for i in range(self.n_seeds):
            rnn_input = embeddings[:, i, :]
            state = self.rnn_cell(rnn_input, state)
        if not self.decoder_fixed_input:
            d_rnn_input = state[0] if self.cell_type == "LSTM" else state
        else:
            d_rnn_input = torch.zeros_like(rnn_input).type_as(rnn_input)
        rnn_outs = []
        # Decoding Steps
        for i in range(n_seqs - self.n_seeds):
            state = self.rnn_cell(d_rnn_input, state)
            # using the output as an input
            if not self.decoder_fixed_input:
                d_rnn_input = state[0] if self.cell_type == "LSTM" else state
                dec_out = self.decoder(d_rnn_input)
            else:
                d_rnn_out = state[0] if self.cell_type == "LSTM" else state
                dec_out = self.decoder(d_rnn_out)

            # residual connection between time step
            if self.residual_step:
                if i == 0:
                    final_out = dec_out + x[:, self.n_seeds-1, :]
                else:
                    final_out = dec_out + rnn_outs[-1]
            else:
                final_out = dec_out
            
            rnn_outs.append(
                final_out
            )
        
        return torch.stack(rnn_outs ,dim=1)

        
    def init_state(self, b_size, x):
        if self.cell_type == "LSTM":
            # type_as lightning thing
            # refer to https://pytorch-lightning.readthedocs.io/en/latest/accelerators/accelerator_prepare.html
            h = torch.zeros(b_size, self.rnn_dim).type_as(x)
            c = torch.zeros(b_size, self.rnn_dim).type_as(x)
            state = (h, c)
        else:
            state = torch.zeros(b_size, self.rnn_dim).type_as(x)
        
        return state


