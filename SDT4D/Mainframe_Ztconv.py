import torch
import torch.nn as nn
###########################
#parallel residual
class MainFrameZT(nn.Module):
    def __init__(
            self,
            img_dim,
            img_time,
            in_channel,
            f_maps=[16, 32, 64],
            input_dropout_rate=0.1,
            num_layers=0,
            bayesian=False
    ):
        super(MainFrameZT, self).__init__()
        self.img_dim = img_dim
        self.img_time = img_time
        self.f_maps = f_maps
        self.bayesian = bayesian

        self.encoders = self.temporalSqueezeZT(
            f_maps=[in_channel] + f_maps
        )

        final_out_channels = 2 if bayesian else in_channel
        self.decoders = self.temporalExcitationZT(
            f_maps=f_maps[::-1] + [final_out_channels]
        )

    def temporalSqueezeZT(self, f_maps, num_layers=0):
        model_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            encoder_layer = SqueezeLayerZT(
                in_channels=f_maps[idx - 1],
                out_channels=f_maps[idx],
                downsample_t=True  # Enable temporal downsampling.
            )
            model_list.append(encoder_layer)
        return model_list

    def temporalExcitationZT(self, f_maps):
        model_list = nn.ModuleList([])
        for idx in range(1, len(f_maps)):
            decoder_layer = ExcitationLayerZT(
                in_channels=f_maps[idx - 1],
                out_channels=f_maps[idx],
                if_up_sample=True
            )
            model_list.append(decoder_layer)
        return model_list

    def process_by_trans(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):
        # x shape: (B, C, Z, T, H, W)
        encoders_features = []
        for encoder in self.encoders:
            before_down, x = encoder(x)
            encoders_features.insert(0, before_down)

        x = self.process_by_trans(x)

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(x, encoder_features)

        return x


class ZTConv(nn.Module):
    """
    Parallel convolution block with optional residual connection.
    Input shape: (B, C, Z, T, H, W)
    """
    def __init__(self, in_channels, out_channels, use_residual=False, align_channels=False):
        super(ZTConv, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv_z = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_thw = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.use_residual = use_residual
        self.align_channels = align_channels

        # Debug/ablation switches (default keeps original behavior).
        # These help diagnose whether spikes come from a specific branch or from summation.
        self.enable_z_path = True
        self.enable_thw_path = True
        self.z_scale = 1.0
        self.thw_scale = 1.0
        self.enable_residual = True
        self.res_scale = 1.0
        self.debug_save = False
        self.last_stats = None

        if use_residual and (in_channels != out_channels):
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x):
        B, C, Z, T, H, W = x.shape

        # Path 1: conv over Z
        out_z = None
        if self.enable_z_path:
            x_z = x.permute(0, 3, 1, 2, 4, 5).reshape(B * T, C, Z, H, W)
            out_z = self.relu(self.conv_z(x_z))
            out_z = out_z.reshape(B, T, -1, Z, H, W).permute(0, 2, 3, 1, 4, 5)

        # Path 2: conv over T-H-W
        out_thw = None
        if self.enable_thw_path:
            x_thw = x.permute(0, 2, 1, 3, 4, 5).reshape(B * Z, C, T, H, W)
            out_thw = self.relu(self.conv_thw(x_thw))
            out_thw = out_thw.reshape(B, Z, -1, T, H, W).permute(0, 2, 1, 3, 4, 5)

        if out_z is None and out_thw is None:
            # Degenerate case: both branches disabled; fall back to zero update.
            out = torch.zeros((B, self.conv_thw.out_channels, Z, T, H, W), device=x.device, dtype=x.dtype)
        elif out_z is None:
            out = self.thw_scale * out_thw
        elif out_thw is None:
            out = self.z_scale * out_z
        else:
            out = self.z_scale * out_z + self.thw_scale * out_thw

        if self.use_residual and self.enable_residual:
            if self.res_conv is not None:
                x_reshaped = x.permute(0, 2, 1, 3, 4, 5).reshape(B * Z, C, T, H, W)
                x_proj = self.res_conv(x_reshaped).reshape(B, Z, -1, T, H, W).permute(0, 2, 1, 3, 4, 5)
            else:
                x_proj = x
            out = out + self.res_scale * x_proj

        if self.debug_save:
            # Save lightweight stats only (avoid storing tensors to prevent memory growth).
            def _stats(t):
                if t is None:
                    return None
                # NOTE: This will sync GPU->CPU when called; enable only for debugging.
                return {
                    "min": float(t.min().detach().cpu().item()),
                    "max": float(t.max().detach().cpu().item()),
                    "mean": float(t.mean().detach().cpu().item()),
                }
            self.last_stats = {
                "z_scale": float(self.z_scale),
                "thw_scale": float(self.thw_scale),
                "res_scale": float(self.res_scale),
                "z": _stats(out_z),
                "thw": _stats(out_thw),
                "out": _stats(out),
            }

        return out




class DoubleZTConv(nn.Sequential):
    """Double convolution block using ZTConv."""
    def __init__(self, in_channels, out_channels, if_encoder):
        super(DoubleZTConv, self).__init__()
        if if_encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.add_module('ZTConv1', ZTConv(conv1_in_channels, conv1_out_channels,use_residual=True))
        self.add_module('ZTConv2', ZTConv(conv2_in_channels, conv2_out_channels,use_residual=True))


class SqueezeLayerZT(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_t=True):
        super(SqueezeLayerZT, self).__init__()
        self.conv_net = DoubleZTConv(
            in_channels=in_channels,
            out_channels=out_channels,
            if_encoder=True
        )
        self.downsample_t = downsample_t
        if downsample_t:
            self.down_sample = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        before_down = self.conv_net(x)
        if self.downsample_t:
            B, C, Z, T, H, W = before_down.shape
            x_down = before_down.permute(0, 2, 1, 3, 4, 5).reshape(B * Z, C, T, H, W)
            x_down = self.down_sample(x_down)
            T_new = x_down.shape[2]
            x = x_down.reshape(B, Z, C, T_new, H, W).permute(0, 2, 1, 3, 4, 5)
        else:
            x = before_down
        return before_down, x


class ExcitationLayerZT(nn.Module):
    def __init__(self, in_channels, out_channels, if_up_sample=True):
        super(ExcitationLayerZT, self).__init__()
        self.conv_net = DoubleZTConv(
            in_channels=in_channels,
            out_channels=out_channels,
            if_encoder=False
        )
        self.if_up_sample = if_up_sample
        # Debug/ablation: scale/disable skip connection addition (default keeps original behavior).
        self.enable_skip = True
        self.skip_scale = 1.0
        if if_up_sample:
            self.up_sample = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(4, 3, 3),
                stride=(2, 1, 1),
                padding=(1, 1, 1))
                # kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x, encoder_features):
        if self.if_up_sample:
            B, C, Z, T, H, W = x.shape
            x_up = x.permute(0, 2, 1, 3, 4, 5).reshape(B * Z, C, T, H, W)
            x_up = self.up_sample(x_up)
            T_new = x_up.shape[2]
            x = x_up.reshape(B, Z, C, T_new, H, W).permute(0, 2, 1, 3, 4, 5)

        if self.enable_skip:
            x = x + self.skip_scale * encoder_features
        x = self.conv_net(x)
        return x
