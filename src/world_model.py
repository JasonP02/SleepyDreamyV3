import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSMWorldModel:
    """
    World model architecture from Dreamerv3
    * 
    """
    def __init__(
            self,
    ):
        self.encoder = ObservationEncoder()

class ObservationEncoder(nn.Module):
    def __init__(self,
                 mlp_config,
                 cnn_config,
                 ):
        super().__init__()
        self.MLP = ObservationMLPEncoder(
            d_in=8,
            d_hidden=mlp_config.d_hidden,
            d_out=mlp_config.d_out
        )
        self.CNN = ObservationCNNEncoder(
            target_size=cnn_config.target_size,
            in_channels=cnn_config.input_channels,
            kernel_size=cnn_config.kernel_size,
            stride=cnn_config.stride,
            padding=cnn_config.padding,
            d_hidden=mlp_config.d_hidden,
            hidden_dim_ratio=mlp_config.hidden_dim_ratio,
            num_layers=cnn_config.num_layers,
            final_feature_size=cnn_config.final_feature_size
        )

        self.latents = mlp_config.d_out
        # Use dynamic calculation based on actual config parameters
        n_channels = int(mlp_config.d_hidden / mlp_config.hidden_dim_ratio)
        cnn_out_features = (n_channels * 2**(cnn_config.num_layers-1)) * cnn_config.final_feature_size**2
        encoder_out = cnn_out_features + mlp_config.d_out # CNN + MLP
        
        # Paper 
        logit_out = self.latents * (self.latents // 16)
        self.logit_layer = nn.Linear(in_features=encoder_out, out_features=logit_out)

    def forward(self, x):
        # x is passed as a dict of ['state', 'pixels']
        image_obs = x['pixels']
        vec_obs = x['state']

        x1 = self.CNN(image_obs)
        x1 = x1.view(x1.size(0), -1) # Flatten all features

        x2 = self.MLP(vec_obs)
        x = torch.cat((x1,x2), dim=1) # Join outputs along feature dimension

        # feed this through a network to get out code * latent size
        x = self.logit_layer(x)
        x = x.view(x.shape[0], self.latents, self.latents//16)

        out =  F.softmax(input=x, dim=2) # output a probability distribution (z)
        return out
        
class ObservationCNNEncoder(nn.Module):
    """
    Observations are compressed dynamically based on config.
    Uses a series of convolutions with doubling channel progression.
    """
    def __init__(self,
                 target_size,
                 in_channels,
                 kernel_size,
                 stride,
                 padding,
                 d_hidden,
                 hidden_dim_ratio=16,
                 num_layers=4,
                 final_feature_size=4
                 ):
        super().__init__()
        self.target_size = target_size
        self.num_layers = num_layers
        self.final_feature_size = final_feature_size

        # Calculate base channel count from hidden dim ratio
        base_channels = int(d_hidden / hidden_dim_ratio)
        
        # Build sequential layers with ReLU activation
        conv_layers = []
        
        for i in range(num_layers):
            if i == 0:
                # First layer: input_channels -> base_channels
                out_ch = base_channels
                conv_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
            else:
                # Subsequent layers: base_channels*2^(i-1) -> base_channels*2^i
                out_ch = base_channels * (2 ** i)
                in_ch = base_channels * (2 ** (i - 1))
                conv_layers.append(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
        
        # Wrap in Sequential for clean forward pass
        self.cnn_blocks = nn.Sequential(*conv_layers)

    def forward(self, x):
        # resize image to target shape
        x = F.interpolate(
            input=x,
            size=self.target_size,
            mode='bilinear',
        )

        # Apply ReLU + convolution sequentially
        x = self.cnn_blocks(x)
        return x

class ObservationMLPEncoder(nn.Module):
    """
    Observations are encoded with 3-layer MLP
    Input state is a vector of size 8
    """
    def __init__(self,
               d_in,
               d_hidden,
               d_out,
               ):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden)
        self.w2 = nn.Linear(d_hidden, d_hidden)
        self.w3 = nn.Linear(d_hidden, d_out)
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.b2 = nn.Parameter(torch.zeros(d_hidden))
        self.b3 = nn.Parameter(torch.zeros(d_out))
        
        self.act = F.relu # TODO review this
    
    def forward(self, x):
        h = self.act(self.w1(x) + self.b1)
        h = self.act(self.w2(h) + self.b2)
        h = self.act(self.w3(h) + self.b3)
        return h



