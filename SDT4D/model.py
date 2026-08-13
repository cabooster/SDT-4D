import torch.nn as nn

from .Mainframe_Ztconv import MainFrameZT, ZTConv
from .swin4d_transformer_ver7 import SwinTransformer4D


class SDT4D(MainFrameZT):
    """4D self-supervised denoising network.

    Tensors use ``(batch, channel, z, time, height, width)`` ordering.
    """

    def __init__(
        self,
        img_dim,
        img_time,
        in_channel=1,
        embedding_dim=64,
        window_size=7,
        num_heads=8,
        hidden_dim=512,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=(8, 16, 32, 64),
        input_dropout_rate=0,
        bayesian=False,
    ):
        super().__init__(
            img_dim,
            img_time,
            in_channel,
            f_maps=list(f_maps),
            input_dropout_rate=input_dropout_rate,
            bayesian=bayesian,
        )
        self.img_time = img_time
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim

        # Kept for checkpoint compatibility with the original implementation.
        self.conv_before_trans = ZTConv(f_maps[-1], embedding_dim)
        self.conv_after_trans = ZTConv(embedding_dim, f_maps[-1])
        self.layers = nn.ModuleList()

        downsampled_time = img_time
        for _ in f_maps:
            downsampled_time = (downsampled_time + 1) // 2
        swin_time_window = max(1, min(4, downsampled_time))
        self.Swin_4D_model = SwinTransformer4D(
            img_size=(5, 128, 128, downsampled_time),
            in_chans=f_maps[-1],
            embed_dim=f_maps[-1],
            window_size=(2, 8, 8, swin_time_window),
            first_window_size=(2, 8, 8, swin_time_window),
            patch_size=(1, 1, 1, 1),
            depths=(2,),
            num_heads=(8,),
        )

    def process_by_trans(self, x):
        x = x.permute(0, 1, 2, 4, 5, 3)
        x = self.Swin_4D_model(x)
        return x.permute(0, 1, 2, 5, 3, 4)
