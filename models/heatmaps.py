import torch
from torch import nn
import torch.nn.functional as F

from .utils import init_state_hm, ConvLSTMCell
from .utils import EncoderBlock, DecoderBlock
from .utils import ResNetEncoder, ResNetDecoder


class StateSpaceHeatmaps(nn.Module):
    """"""
    def __init__(
        self,
        in_ch=17, # Number of input channels
        enc_layers_ch=[32, 64],
        # dec_layers_ch=[32, 17],
        n_cells=1,
        rnn_ch=64, # Channels for convLSTM
        rnn_kernel_size=(3,3),
        residual_step=True,
        n_seeds=10,
        decoder_fixed_input=True,
        use_batched_input=False, # Check forward method for explanation
        **kwargs
    ):
        super().__init__()

        if not decoder_fixed_input:
            assert enc_layers_ch[-1] == rnn_ch, "To have the same input channels"\
            " as output, rnn input and output should have the same number of channels"\
        # ResNet Encoder
        self.encoder = ResNetEncoder(EncoderBlock, enc_layers_ch)
        # ConvLSTMS
        rnn_cells = []
        for i in range(n_cells):
            in_size = enc_layers_ch[-1] if i == 0 else rnn_ch
            rnn_cells.append(
                ConvLSTMCell(
                    input_dim=in_size,
                    hidden_dim=rnn_ch,
                    kernel_size=rnn_kernel_size,
            )
        )
        self.rnn_cells = nn.ModuleList(rnn_cells)
        # ResNet Decoder
        # Mirrored version of encoder
        if len(enc_layers_ch) > 1:
            dec_layers_ch = enc_layers_ch[::-1][1:]
        else:
            dec_layers_ch = []
        
        dec_layers_ch.append(in_ch)

        self.decoder = ResNetDecoder(DecoderBlock, dec_layers_ch, rnn_ch)
        self.rnn_ch = rnn_ch
        self.n_cells = n_cells
        self.residual_step = residual_step
        self.n_seeds = n_seeds
        self.decoder_fixed_input = decoder_fixed_input
        self.use_batched_input = use_batched_input
        
    def forward(self, x):
        b_size, n_seqs, n_joints, height, width = x.shape
        # nn.conv2d expects 4D batch tensor but we have 5D tensor
        # using for loop over batch will be a bit slower at the same reshaping tensor
        # passing to encoder for a larger batch size will not fit gpu memory
        if self.use_batched_input:
            embeddings = self.encoder(torch.flatten(x[:, 0:self.n_seeds, :, :, :], start_dim=0, end_dim=1))
            embeddings = embeddings.reshape(
                b_size,
                self.n_seeds,
                embeddings.shape[1],
                embeddings.shape[2],
                embeddings.shape[3]
            )
        else:
            embeddings = torch.stack([
                self.encoder(x[:,i,:,:,:]) for i in range(self.n_seeds)
            ], dim=1)
        
        # initial input of rnn
        states = init_state_hm(
            n_cells=self.n_cells,
            rnn_ch=self.rnn_ch,
            x=embeddings
        )
        # Encoding Steps
        for i in range(self.n_seeds):
            rnn_input = embeddings[:, i, :, :, :]
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0]
        
        if self.decoder_fixed_input:
            rnn_input = torch.zeros_like(embeddings[:, 0, :, :, :]).type_as(embeddings)
        
        outs = []
        # Decoding Steps
        for i in range(n_seqs - self.n_seeds):
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0]
            dec_out = self.decoder(rnn_input)
            # Residual connection between time step
            if self.residual_step:
                if i == 0:
                    final_out = dec_out + x[:, self.n_seeds-1, :, :, :]
                else:
                    final_out = dec_out + outs[-1]
                
            outs.append(
                final_out
            )
            if self.decoder_fixed_input:
                rnn_input = torch.zeros_like(embeddings[:, 0, :, :, :]).type_as(embeddings)
        
        return torch.stack(outs, dim=1)


class AutoregressiveHeatmaps(nn.Module):
    """"""
    def __init__(
        self,
        in_ch=17, # Number of input channels
        enc_layers_ch=[32, 64],
        n_cells=1,
        rnn_ch=64, # Channels for convLSTM
        rnn_kernel_size=(3,3),
        residual_step=True,
        n_seeds=10,
        use_batched_input=False, # Check forward method for explanation
        **kwargs
    ):
        super().__init__()
        # ResNet Encoder
        self.encoder = ResNetEncoder(EncoderBlock, enc_layers_ch)
        # ConvLSTMS
        rnn_cells = []
        for i in range(n_cells):
            in_size = enc_layers_ch[-1] if i == 0 else rnn_ch
            rnn_cells.append(
                ConvLSTMCell(
                    input_dim=in_size,
                    hidden_dim=rnn_ch,
                    kernel_size=rnn_kernel_size,
            )
        )
        self.rnn_cells = nn.ModuleList(rnn_cells)
        # ResNet Decoder
        # Mirrored version of encoder
        if len(enc_layers_ch) > 1:
            dec_layers_ch = enc_layers_ch[::-1][1:]
        else:
            dec_layers_ch = []
        
        dec_layers_ch.append(in_ch)
        self.decoder = ResNetDecoder(DecoderBlock, dec_layers_ch, rnn_ch)

        self.rnn_ch = rnn_ch
        self.n_cells = n_cells
        self.residual_step = residual_step
        self.n_seeds = n_seeds
        self.use_batched_input = use_batched_input
    
    def forward(self, x):
        b_size, n_seqs, n_joints, height, width = x.shape
        # nn.conv2d expects 4D batch tensor but we have 5D tensor
        # using for loop over batch will be a bit slower at the same reshaping tensor
        # passing to encoder for a larger batch size will not fit gpu memory
        if self.use_batched_input:
            embeddings = self.encoder(torch.flatten(x[:, 0:self.n_seeds, :, :, :], start_dim=0, end_dim=1))
            embeddings = embeddings.reshape(
                b_size,
                self.n_seeds,
                embeddings.shape[1],
                embeddings.shape[2],
                embeddings.shape[3]
            )
        else:
            embeddings = torch.stack([
                self.encoder(x[:,i,:,:,:]) for i in range(self.n_seeds)
            ], dim=1)
        
        # initial input of rnn
        states = init_state_hm(
            n_cells=self.n_cells,
            rnn_ch=self.rnn_ch,
            x=embeddings
        )
        # Encoding Steps
        for i in range(self.n_seeds):
            rnn_input = embeddings[:,i, :, :, :]
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0]
        
        dec_out = self.decoder(rnn_input)
        rnn_input = self.encoder(dec_out)

        outs = []
        # Decoding Steps
        for i in range(n_seqs - self.n_seeds):
            for j, rnn_cell in enumerate(self.rnn_cells):
                states[j] = rnn_cell(rnn_input, states[j])
                rnn_input = states[j][0]
            
            dec_out = self.decoder(rnn_input)
            # Residual connection between time step
            if self.residual_step:
                if i == 0:
                    final_out = dec_out + x[:, self.n_seeds-1, :, :, :]
                else:
                    final_out = dec_out + outs[-1]
            else:
                final_out = dec_out
            
            outs.append(
                final_out
            )
            rnn_input = self.encoder(dec_out)

        return torch.stack(outs, dim=1)