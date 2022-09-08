import torch
from torch import nn
import torch.nn.functional as F

from .utils import get_dropout, get_act
from .utils import LinearBlock, init_state


class StateSpace2dJoints(nn.Module):
    """"""
    def __init__(
        self,
        input_dim=34,
        encoder_layers=[128],
        cell_type="LSTM",
        n_cells=1,
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
        rnn_cells = []
        for i in range(n_cells):
            in_size = encoder_layers[-1] if i == 0 else rnn_dim
            rnn_cells.append(
                rnn_cell(input_size=in_size, hidden_size=rnn_dim)
            )
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.decoder = LinearBlock(rnn_dim, sizes=[input_dim], act=None)
        self.rnn_dim = rnn_dim
        self.n_cells = n_cells
        self.cell_type = cell_type
        self.residual_step = residual_step
        self.n_seeds = n_seeds
        self.decoder_fixed_input = decoder_fixed_input
    

    def forward(self, x):
        b_size, n_seqs, n_joints = x.shape
        # initial input of rnn
        states = init_state(
            cell_type=self.cell_type,
            n_cells=self.n_cells,
            rnn_dim=self.rnn_dim,
            b_size=b_size,
            x=x
        )
        embeddings = self.encoder(x[:, 0:self.n_seeds, :])
        # Encoding Steps
        for i in range(self.n_seeds):
            rnn_input = embeddings[:, i, :]
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0] if self.cell_type == "LSTM" else states[j]
        if self.decoder_fixed_input:
            rnn_input = torch.zeros_like(embeddings[:, 0, :]).type_as(embeddings[:, 0, :])
        
        outs = []
        # Decoding Steps
        for i in range(n_seqs - self.n_seeds):
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0] if self.cell_type == "LSTM" else states[j]
            dec_out = self.decoder(rnn_input)
            # Residual connection between time step
            if self.residual_step:
                if i == 0:
                    final_out = dec_out + x[:, self.n_seeds-1, :]
                else:
                    final_out = dec_out + outs[-1]
            else:
                final_out = dec_out
            
            outs.append(
                final_out
            )
            

            if self.decoder_fixed_input:
                rnn_input = torch.zeros_like(embeddings[:, 0, :]).type_as(embeddings[:, 0, :])
        
        return torch.stack(outs, dim=1)


class Autoregressive2dJoints(nn.Module):
    """Model 2"""
    def __init__(
        self,
        input_dim=34,
        encoder_layers=[128],
        cell_type="LSTM",
        n_cells=1,
        rnn_dim=64,
        residual_step=True,
        n_seeds=10,
        **kwargs
    ):
        super().__init__()
        # Linear encoder
        self.encoder = LinearBlock(input_dim, encoder_layers, **kwargs)
        rnn_cell = nn.LSTMCell if cell_type=="LSTM" else nn.GRUCell
        rnn_cells = []
        for i in range(n_cells):
            in_size = encoder_layers[-1] if i == 0 else rnn_dim
            rnn_cells.append(
                rnn_cell(input_size=in_size, hidden_size=rnn_dim)
            )
        self.rnn_cells = nn.ModuleList(rnn_cells)
        # Linear decoder
        self.decoder = LinearBlock(rnn_dim, sizes=[input_dim], act=None)
        self.cell_type = cell_type
        self.n_cells = n_cells
        self.rnn_dim = rnn_dim
        self.residual_step = residual_step
        self.n_seeds = n_seeds
    
    def forward(self, x):
        b_size, n_seqs, n_joints = x.shape
        # initial input of rnn
        states = init_state(
            cell_type=self.cell_type,
            n_cells=self.n_cells,
            rnn_dim=self.rnn_dim,
            b_size=b_size,
            x=x
        )

        embeddings = self.encoder(x[:, 0:self.n_seeds, :])
        # Encoding Steps
        for i in range(self.n_seeds):
            rnn_input = embeddings[:, i, :]
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0] if self.cell_type == "LSTM" else states[j]
        
        dec_out = self.decoder(rnn_input)
        rnn_input = self.encoder(dec_out)

        outs = []
        # Decoding Steps
        for i in range(n_seqs - self.n_seeds):
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0] if self.cell_type == "LSTM" else states[j]
            
            dec_out = self.decoder(rnn_input)
            # Residual connection between time step
            if self.residual_step:
                if i == 0:
                    final_out = dec_out + x[:, self.n_seeds-1, :]
                else:
                    final_out = dec_out + outs[-1]
            else:
                final_out = dec_out
            
            outs.append(
                final_out
            )
            rnn_input = self.encoder(dec_out)

        return torch.stack(outs, dim=1)