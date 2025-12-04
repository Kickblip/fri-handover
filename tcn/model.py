import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """
    Removes the padding from the end of the sequence to ensure causality.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Direct Prediction TCN.
    Encodes history -> takes last timestep -> projects to all future frames at once.
    """
    def __init__(self, in_dim, num_channels, kernel_size=2, dropout=0.2, future_frames=10, out_dim=63):
        super(TemporalConvNet, self).__init__()
        self.future_frames = future_frames
        self.out_dim = out_dim
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = in_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding is calculated to maintain sequence length (causal)
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
        # --- DIRECT PREDICTION HEAD ---
        # Projects the hidden state of the LAST timestep to (future_frames * out_dim)
        self.decoder = nn.Linear(num_channels[-1], future_frames * out_dim)

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len, in_dim]
        Returns:
            [Batch, future_frames, out_dim]
        """
        # Permute to [B, C, L] for Conv1d
        x = x.permute(0, 2, 1) 
        
        # Pass through TCN backbone
        y = self.network(x) # Output: [B, C_out, L]
        
        # Take the last timestep: [B, C_out]
        # This vector contains the compressed history
        last_timestep = y[:, :, -1] 
        
        # Project to flat prediction vector
        prediction_flat = self.decoder(last_timestep)
        
        # Reshape to [B, future_frames, output_dim]
        prediction = prediction_flat.view(-1, self.future_frames, self.out_dim)
        
        return prediction