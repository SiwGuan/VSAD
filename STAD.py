import nntplib

from modules import *
import torch.nn.functional as F


class ST_encoder(nn.Module):
    def __init__(self, n_features, d_model, window):
        super(ST_encoder, self).__init__()
        self.linear = nn.Linear(n_features, d_model)
        self.tcna = TemporalConvNet(num_inputs=d_model, num_channels=[d_model], d_model=d_model, kernel_size=2,
                                   attention=True)
        self.tcnb = TemporalConvNet(num_inputs=d_model, num_channels=[d_model], d_model=d_model, kernel_size=2,
                                   attention=True)
        self.sgnn = SGNN(d_model)
        self.position = PositionalEncoding(d_model=d_model, max_len=window)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=256, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 2)


        decoder_layers1 = nn.TransformerDecoderLayer(d_model=d_model, nhead=1, dim_feedforward=256, dropout=0.3)
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = nn.TransformerDecoderLayer(d_model=d_model, nhead=1, dim_feedforward=256, dropout=0.3)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, 1)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x_a = self.tcna(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_b = self.tcna(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.tanh(x_b)*torch.sigmoid(x_a)
        x = self.sgnn(x)
        x = self.position(x)
        x = self.transformer_encoder(x.permute(1, 0, 2))
        mu = self.transformer_decoder1(x[-1, :, :].unsqueeze(0), x)
        logvar = self.transformer_decoder1(x[-1, :, :].unsqueeze(0), x)
        return mu.squeeze(), logvar.squeeze()


class STAD(nn.Module):
    def __init__(self, n_features, window, d_model, n_latent=16):
        super(STAD, self).__init__()
        self.window = window
        self.d_model = d_model
        self.n_latent = n_latent
        self.n_hidden = n_latent * 4
        self.n_features = n_features
        self.linear = nn.Linear(n_features, d_model)
        self.ST_encoder = ST_encoder(d_model, d_model, window)
        self.linear2 = nn.Linear(d_model,n_features)
        self.linear3 = nn.Linear(d_model,n_features)
        self.end = nn.Sequential(

            # nn.Linear(d_model, d_model), nn.ReLU(),
            # nn.Linear(d_model, n_features), nn.ReLU(),
            nn.Linear(n_features, n_features), nn.Sigmoid(),
        )

    def forward(self, x):
        x = F.relu(self.linear(x))
        mu, logvar = self.ST_encoder(x)
        mu = self.linear2(mu)
        logvar = self.linear3(logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x = self.end(z.squeeze())
        return x, mu, logvar
