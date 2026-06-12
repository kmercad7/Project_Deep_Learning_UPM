from dataclasses import dataclass, field
from Mamba_blocks import MixerModel
import torch.nn.functional as F
import os, json
import torch
import torch.nn as nn

@dataclass
class HARMambaConfig:
    d_model: int = 160
    num_sensor_features: int = 27
    d_intermediate: int = 320
    n_layer: int = 7
    ssm_cfg: dict = field(default_factory=lambda: {"expand": 2, "layer": "Mamba2", "headdim": 8, "d_ssm": 192, "dt_min": 0.001, "dt_max": 0.1,})
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    
class MambaEncoderModel(nn.Module):
    def __init__(self, config, initializer_cfg=None, device=None, dtype=None) -> None:
        super().__init__()
        
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": device, "dtype": dtype}

        self.kernel_size = 9
        self.stride = 1
        self.padding = 4
        
        # HAR INPUT AND BLOCKS
        # CONV1D -> BATCHNORM -> GELU -> BLURPOOL1D -> Backbone (MAMBA) -> Classifier
        self.convolutional_input = nn.Conv1d(
            in_channels=config.num_sensor_features, 
            out_channels=d_model, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            padding_mode="reflect"
        )
        self.batch_layer = nn.BatchNorm1d(d_model) 
        self.act_function = nn.GELU()

        self.backbone = MixerModel(
            preprocessed_features=d_model, 
            d_model=d_model, 
            n_layer=n_layer, 
            d_intermediate=d_intermediate, 
            ssm_cfg=ssm_cfg, 
            rms_norm=rms_norm, 
            initializer_cfg=initializer_cfg, 
            fused_add_norm=fused_add_norm, 
            residual_in_fp32=residual_in_fp32, 
            **factory_kwargs
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)       
        
    def forward(self, input_sensor1, inference_params=None, **mixer_kwargs) -> torch.tensor:
        '''
        Sensor Input -> [Batch_size, Length_window, num_sensor_features]
        Output -> hidden_states = [B, L, d_model]
        '''
        x = input_sensor1.transpose(1, 2).contiguous()  # [B, L, Ft] -> [B, Ft, L]
        x = self.convolutional_input(x) # [B, d_model, L]
        x = self.batch_layer(x) # [B, d_model, L]
        x = self.act_function(x) # [B, d_model, L]
        #x = self.lpf_downsampling(x) # [B, d_model, 150] (stride 1)
        x = x.transpose(1, 2)  # [B, d_model, L] -> [B, L, d_model]
        
        hidden_states = self.backbone(x, inference_params=inference_params, **mixer_kwargs) # [B, L, d_model]
        return hidden_states

class DecoderModel(nn.Module):
    def __init__(self, config, device=None, dtype=None) -> None:
        super().__init__()

        d_model = config.d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.decoder = nn.Sequential(
        nn.Linear(d_model, d_model, **factory_kwargs),
        nn.GELU(),
        nn.LayerNorm(d_model, **factory_kwargs),
        nn.Linear(d_model, config.num_sensor_features, **factory_kwargs)
        )
          
    def forward(self, hidden_states) -> torch.tensor:
        '''
        Input -> hidden_states = [B, L, d_model]a
        Output -> Reconstructed signal [Batch_size, L, num_sensor_features]
        '''
        decoder = self.decoder(hidden_states) # [B, L, num_sensor_features]
        return decoder   

class MambaAutoencoderModel(nn.Module):
    def __init__(self, config, initializer_cfg=None, device=None, dtype=None) -> None:
        super().__init__()
        
        self.encoder = MambaEncoderModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype
        )

        self.decoder = DecoderModel(
            config=config,
            device=device,
            dtype=dtype
        )

    def forward(self, input_sensor1, inference_params=None, **mixer_kwargs):
        hidden_states = self.encoder(input_sensor1, inference_params=inference_params, **mixer_kwargs)
        x_rec = self.decoder(hidden_states)
        return x_rec
        
class MambaDownstreamClassifier(nn.Module):
    def __init__(self, config, num_classes: int, initializer_cfg=None, device=None, dtype=None) -> None:
        super().__init__()
        
        self.encoder = MambaEncoderModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype
        )
        
        d_model = config.d_model
        factory_kwargs = {"device": device, "dtype": dtype}

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model, **factory_kwargs),
            nn.Linear(d_model, d_model, **factory_kwargs),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes, **factory_kwargs)
        )

    def forward(self, x):
        hidden_states = self.encoder(x) # [B, L, d_model]
        pooled_features = hidden_states.mean(dim=1)   # [B, d_model]
        logits = self.classifier(pooled_features)
        return logits



        