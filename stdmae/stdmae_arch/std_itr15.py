import torch
from torch import nn

from .mask import Mask
from .graphwavenet import GraphWaveNet_itr15


class STD_ITR15(nn.Module):
    """Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting"""

    def __init__(self, dataset_name, pre_trained_tmae_path,pre_trained_smae_path, mask_args, backend_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.pre_trained_smae_path = pre_trained_smae_path
        # iniitalize 
        self.tmae = Mask(**mask_args)
        self.smae = Mask(**mask_args)

        self.backend = GraphWaveNet_itr15(**backend_args)

        print("self.backend",self.backend)

        # load pre-trained model
        self.load_pre_trained_model()


    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters

        # 原始东西  20240604
        # checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        checkpoint_dict = torch.load(self.pre_trained_tmae_path, map_location=torch.device('cpu'))

        self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])

        # 原始东西  20240604
        # checkpoint_dict = torch.load(self.pre_trained_smae_path)
        checkpoint_dict = torch.load(self.pre_trained_smae_path,map_location=torch.device('cpu'))

        self.smae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STDMAE.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """

        # reshape
        short_term_history = history_data     # [B, L, N, 1]



        batch_size, _, num_nodes, _ = history_data.shape



        hidden_states_t = self.tmae(long_history_data[..., [0]])
        hidden_states_s = self.smae(long_history_data[..., [0]])
        hidden_states=torch.cat((hidden_states_t,hidden_states_s),-1)
        
        # enhance
        out_len=1
        hidden_states = hidden_states[:, :, -out_len, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1)

        return y_hat

