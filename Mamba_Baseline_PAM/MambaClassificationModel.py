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

def _binomial_1d(filt_size: int) -> torch.tensor:
    '''
    Return normalized 1D binomial (approx Gaussian) kernel.
    '''
    if filt_size == 3:
        coeffs = [1, 2, 1]
    elif filt_size == 5:
        coeffs = [1, 4, 6, 4, 1]
    elif filt_size == 7:
        coeffs = [1, 6, 15, 20, 15, 6, 1]
    else:
        raise ValueError("filt_size must be one of {3,5,7}")
    k = torch.tensor(coeffs, dtype=torch.float32)
    k = k / k.sum()
    return k  # shape [K]

class BlurPool1D(nn.Module):
    '''
    Anti-aliased downsampling for 1D: reflect-pad -> depthwise blur -> stride.
    x: [B, C, L] -> [B, C, L_out]
    '''
    def __init__(self, channels: int, filt_size: int = 5, stride: int = 3, padding_mode: str = "reflect"):
        super().__init__()
        self.stride = stride
        self.pad = filt_size // 2
        self.padding_mode = padding_mode

        k = _binomial_1d(filt_size).view(1, 1, -1)         # [1,1,K]
        # Depthwise conv with fixed weights
        self.register_buffer("kernel", k)                  # not a Parameter; no grads
        self.channels = channels
        self.filt_size = filt_size

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: [B, C, L]
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode=self.padding_mode)
        # Depthwise conv: groups = channels, stride = downsampling factor
        weight = self.kernel.expand(self.channels, 1, self.filt_size)  # [C,1,K]
        x = F.conv1d(x, weight, stride=self.stride, groups=self.channels)
        return x
    
class MambaClassificationModel(nn.Module):
    def __init__(self, config, num_classes: int, initializer_cfg=None, device=None, dtype=None) -> None:
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
        # CONV1D -> GELU -> BLURPOOL1D -> Backbone (MAMBA) -> Classifier
        
        self.convolutional_input = nn.Conv1d(
            in_channels=config.num_sensor_features, 
            out_channels=d_model, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            padding_mode="reflect"
        )
        self.batch_layer = nn.BatchNorm1d(d_model) #####
        self.act_function = nn.GELU()
        #self.lpf_downsampling = BlurPool1D(channels=d_model, filt_size=5, stride = 3, padding_mode="reflect")

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

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model, **factory_kwargs),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes, **factory_kwargs)
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)       
        
    def forward(self, input_sensor1, inference_params=None, **mixer_kwargs) -> torch.tensor:
        '''
        Sensor Input -> [Batch_size, Length_window, num_sensor_features]
        Output -> Logits [Batch_size, num_classes]
        '''
        x = input_sensor1.transpose(1, 2).contiguous()  # [B, L, F] -> [B, F, L]
        x = self.convolutional_input(x)
        x = self.batch_layer(x) ######      
        x = self.act_function(x)
        #x = self.lpf_downsampling(x)
        x = x.transpose(1, 2).contiguous()  # [B, F, L] -> [B, L, F]

        hidden_states = self.backbone(x, inference_params=inference_params, **mixer_kwargs)
        pooled_features = hidden_states.mean(dim=1)  # mean pooling over timesteps: [B, L, F] -> [B, F]
        logits = self.classifier(pooled_features)
        return logits

        
        
        
