import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.blocks.dynunet_block import (
    UnetOutBlock,
    UnetResBlock,
    UnetUpBlock,
    get_conv_layer,
)
from monai.networks.blocks.squeeze_and_excitation import ResidualSELayer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.dynunet import DynUNet
############################################################


class ScharrEdge(nn.Module):
    """3D Scharr 边缘检测算子"""

    def __init__(self, dim):
        super().__init__()

        scharr_x = torch.tensor(
            [
                [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                [[-6, 0, 6], [-20, 0, 20], [-6, 0, 6]],
                [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
            ],
            dtype=torch.float32,
        )
        scharr_y = torch.tensor(
            [
                [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                [[-6, -20, -6], [0, 0, 0], [6, 20, 6]],
                [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
            ],
            dtype=torch.float32,
        )
        scharr_z = torch.tensor(
            [
                [[-3, -6, -3], [-10, -20, -10], [-3, -6, -3]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[3, 6, 3], [10, 20, 10], [3, 6, 3]],
            ],
            dtype=torch.float32,
        )

        self.weight_x = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_y = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_z = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)

        self.weight_x.weight.data = scharr_x.view(1, 1, 3, 3, 3).repeat(dim, 1, 1, 1, 1)
        self.weight_y.weight.data = scharr_y.view(1, 1, 3, 3, 3).repeat(dim, 1, 1, 1, 1)
        self.weight_z.weight.data = scharr_z.view(1, 1, 3, 3, 3).repeat(dim, 1, 1, 1, 1)

    def forward(self, x):
        """传统 PyTorch 实现"""
        # 使用分组卷积实现 Scharr 算子
        gx = self.weight_x(x)
        gy = self.weight_y(x)
        gz = self.weight_z(x)

        edge = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-6)
        return edge


class MultiScaleGaussian(nn.Module):
    def __init__(self, dim, sizes, sigmas, padding_mode="replicate"):
        super().__init__()
        assert len(sizes) == len(sigmas) and len(sizes) > 0
        for s in sizes:
            assert s % 2 == 1, f"Gaussian kernel size must be odd, got {s}"
        for sg in sigmas:
            assert float(sg) > 0, f"sigma must be > 0, got {sg}"

        self.dim = dim
        self.filters = nn.ModuleList()
        for size, sigma in zip(sizes, sigmas):
            conv = nn.Conv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=size,
                padding=size // 2,
                groups=dim,
                bias=False,
                padding_mode=padding_mode,
            )
            with torch.no_grad():
                k = self.build_kernel(size, float(sigma))
                k = k.to(dtype=conv.weight.dtype, device=conv.weight.device)
                conv.weight.copy_(k.repeat(dim, 1, 1, 1, 1))
            self.filters.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [f(x) for f in self.filters]
        return torch.stack(outs, dim=0).mean(0)

    @staticmethod
    def build_kernel(size: int, sigma: float) -> torch.Tensor:
        r = size // 2
        coords = torch.arange(-r, r + 1, dtype=torch.float32)
        z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
        sq = x**2 + y**2 + z**2
        kernel = torch.exp(-0.5 * sq / (sigma**2))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel.unsqueeze(0).unsqueeze(0)


class PEF(nn.Module):
    def __init__(self, dim, out_dim, norm_layer, act_layer):
        super().__init__()
        self.gaussian = MultiScaleGaussian(dim, sizes=[3, 5, 7], sigmas=[0.8, 1.2, 1.4])
        self.scharr = ScharrEdge(dim)
        self.norm = get_norm_layer(norm_layer, spatial_dims=3, channels=dim * 3)
        self.act = get_act_layer(act_layer)
        self.eca = ResidualSELayer(3, dim * 3, acti_type_2="sigmoid")

        self.fuse = get_conv_layer(
            3, dim * 3, out_dim, kernel_size=1, act=act_layer, norm=norm_layer
        )

    def forward(self, x):
        g = self.gaussian(x)
        e = self.scharr(x)
        fuse = self.norm(self.act(torch.cat([x, g, e], dim=1)))
        out = self.eca(fuse)
        return self.fuse(out)


################## Attn


class SparseAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attn_stride=4, num_heads=8):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=attn_stride)
        # Linear transformations for query, key, and value
        self.query = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv_trans = nn.ConvTranspose3d(
            out_channels,
            out_channels,
            kernel_size=attn_stride,
            stride=attn_stride,
            groups=out_channels,
        )

    def forward(self, x):
        res = x
        x = self.sparse_sampler(x)
        # Query, key, and value projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # Reshape for multi-head attention
        B, C, W, H, D = x.size()
        q = q.view(B, self.num_heads, -1, W * H * D).permute(0, 1, 3, 2).contiguous()
        k = k.view(B, self.num_heads, -1, W * H * D).permute(0, 1, 3, 2).contiguous()
        v = v.view(B, self.num_heads, -1, W * H * D).permute(0, 1, 3, 2).contiguous()
        context = F.scaled_dot_product_attention(
            F.normalize(q, dim=-1), F.normalize(k, dim=-1), v
        )

        # Reshape and concatenate multi-head outputs
        context = context.permute(0, 1, 3, 2).contiguous().view(B, C, W, H, D)
        context = self.conv_trans(context)
        return context + res


################## SegHead
class SDHead(nn.Module):
    def __init__(self, top_c, bottom_c, num_classes, act_layer, norm_layer):
        super(SDHead, self).__init__()

        self.bottom_conv = nn.Conv3d(bottom_c, 1, 3, 1, 1)

        self.ff_conv = nn.Conv3d(  # Depthwise
            in_channels=top_c,
            out_channels=top_c,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=top_c,
        )
        self.bf_conv = nn.Conv3d(  # Depthwise
            in_channels=top_c,
            out_channels=top_c,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=top_c,
        )
        # 前后流互注意力增强
        self.cross_attention = CrossAttentionBlock(
            hidden_size=top_c,
            num_heads=4,
            dropout_rate=0.0,
            use_flash_attention=True,
        )

        self.downsample_factor = 8
        self.conv_trans = nn.ConvTranspose3d(
            top_c,
            top_c,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
            groups=top_c,
        )

        self.se = ResidualSELayer(3, top_c * 3, acti_type_2="sigmoid")

        self.seg_head = UnetOutBlock(3, top_c * 3, num_classes)

    def forward(self, feat, bottom):
        B, C, D, H, W = feat.size()
        bottom = F.sigmoid(
            F.interpolate(
                self.bottom_conv(bottom),
                size=(D, H, W),
                mode="trilinear",
                align_corners=True,
            )
        )

        ff_feat = self.ff_conv(feat * bottom)
        bf_feat = self.bf_conv(feat * (1 - bottom))

        ff_feat_down = F.avg_pool3d(
            ff_feat, kernel_size=1, stride=self.downsample_factor
        )
        bf_feat_down = F.avg_pool3d(
            bf_feat, kernel_size=1, stride=self.downsample_factor
        )

        ff_feat_seq = ff_feat_down.view(B, C, -1).transpose(1, 2)
        bf_feat_seq = bf_feat_down.view(B, C, -1).transpose(1, 2)

        # 交叉注意力增强
        ff_enhanced_seq = self.cross_attention(ff_feat_seq, bf_feat_seq)
        bf_enhanced_seq = self.cross_attention(bf_feat_seq, ff_feat_seq)

        # 恢复3D特征图格式并上采样回原始尺寸
        target_d, target_h, target_w = (
            D // self.downsample_factor,
            H // self.downsample_factor,
            W // self.downsample_factor,
        )
        ff_enhanced_down = ff_enhanced_seq.transpose(1, 2).view(
            B, C, target_d, target_h, target_w
        )
        bf_enhanced_down = bf_enhanced_seq.transpose(1, 2).view(
            B, C, target_d, target_h, target_w
        )

        ff_enhanced = self.conv_trans(ff_enhanced_down) + ff_feat
        bf_enhanced = self.conv_trans(bf_enhanced_down) + bf_feat

        fusion = torch.cat((ff_enhanced, bf_enhanced, feat), dim=1)

        seg_res = self.seg_head(self.se(fusion))
        return seg_res


##############################################################


class PRODNet(nn.Module):
    def __init__(
        self,
        dim_in=1,
        num_classes=1000,
        depths=[1, 2, 4, 2],
        embed_dims=[24, 48, 96, 192],
        drop=0.0,
        norm_layer=("instance", {"affine": True}),
        act_layer="GELU",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.skip = get_conv_layer(
            3,
            dim_in,
            embed_dims[0],
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            dropout=drop,
            conv_only=False,
            norm=norm_layer,
            act=act_layer,
        )
        self.stage0 = get_conv_layer(
            3,
            dim_in,
            embed_dims[0],
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            dropout=drop,
            conv_only=False,
            norm=norm_layer,
            act=act_layer,
        )
        self.skip_down = get_conv_layer(
            3,
            embed_dims[0],
            embed_dims[3],
            kernel_size=(4, 8, 8),
            stride=(4, 8, 8),
            dropout=drop,
            conv_only=False,
            norm=norm_layer,
            act=act_layer,
        )

        self.stages = nn.ModuleList()
        self.downlayers = nn.ModuleList()
        self.translayersdown = nn.ModuleList()
        self.translayersup = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.translayersdown.append(
            SparseAttention(
                in_channels=embed_dims[0],
                out_channels=embed_dims[0],
                num_heads=4,
                attn_stride=2,
            ),
        )
        self.translayersup.append(
            SparseAttention(
                in_channels=embed_dims[3],
                out_channels=embed_dims[3],
                num_heads=4,
                attn_stride=2,
            )
        )

        n = len(depths)
        for i in range(n):
            # ---------- stages ----------
            self.stages.append(
                nn.Sequential(
                    *[
                        UnetResBlock(
                            spatial_dims=3,
                            in_channels=embed_dims[i],
                            out_channels=embed_dims[i],
                            kernel_size=3,
                            stride=1,
                            norm_name=norm_layer,
                            act_name=act_layer,
                        )
                        for _ in range(depths[i])
                    ]
                )
            )

            # 只有在不是最后一层时，才需要 down 相关的层
            if i < n - 1:
                # ---------- downlayers ----------
                k_down = s_down = (1, 2, 2) if i == 0 else (2, 2, 2)
                self.downlayers.append(
                    get_conv_layer(
                        3,
                        embed_dims[i],
                        embed_dims[i + 1],
                        kernel_size=k_down,
                        stride=s_down,
                        dropout=drop,
                        conv_only=False,
                        norm=norm_layer,
                        act=act_layer,
                    )
                )

                # ---------- translayersdown ----------
                j = i + 1
                down_size = (1, 2, 2) if j == 1 else (2, 2, 2)
                self.translayersdown.append(
                    nn.Sequential(
                        get_conv_layer(
                            3,
                            embed_dims[j - 1],
                            embed_dims[j],
                            kernel_size=down_size,
                            stride=down_size,
                            dropout=drop,
                            conv_only=False,
                            norm=norm_layer,
                            act=act_layer,
                        ),
                        SparseAttention(
                            in_channels=embed_dims[j],
                            out_channels=embed_dims[j],
                            num_heads=4,
                            attn_stride=2,
                        ),
                    )
                )

                # ---------- translayersup ----------
                up_size = (1, 2, 2) if j == n - 1 else (2, 2, 2)
                self.translayersup.append(
                    get_conv_layer(
                        spatial_dims=3,
                        in_channels=embed_dims[n - j],
                        out_channels=embed_dims[n - j - 1],
                        kernel_size=up_size,
                        stride=up_size,
                        norm=norm_layer,
                        act=act_layer,
                        conv_only=False,
                        is_transposed=True,
                    )
                )
            # ---------- fusion ----------
            self.fusion.append(
                PEF(
                    dim=3 * embed_dims[i],
                    out_dim=embed_dims[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            )

        self.up3 = UnetUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[3],
            out_channels=embed_dims[2],
            kernel_size=3,
            stride=2,
            upsample_kernel_size=2,
            norm_name=norm_layer,
            act_name=act_layer,
        )
        self.up2 = UnetUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[2],
            out_channels=embed_dims[1],
            kernel_size=3,
            stride=2,
            upsample_kernel_size=2,
            norm_name=norm_layer,
            act_name=act_layer,
        )
        self.up1 = UnetUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[1],
            out_channels=embed_dims[0],
            kernel_size=3,
            stride=(1, 2, 2),
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_layer,
            act_name=act_layer,
        )
        self.output = SDHead(
            top_c=embed_dims[0],
            bottom_c=2 * embed_dims[0],
            act_layer=act_layer,
            norm_layer=norm_layer,
            num_classes=num_classes,
        )
        self.apply(DynUNet.initialize_weights)

    def forward(self, x):
        # ---------- 0. shortcuts ----------
        skip_in = self.skip(x)
        x0 = self.stage0(x)  # 最浅层特征 (与最后拼接)

        # ---------- 1. encoder 主干 ----------
        x_feats = []
        x = x0
        for i in range(len(self.stages)):
            if i > 0:  # 第0层之前不需要down
                x = self.downlayers[i - 1](x)
            x = self.stages[i](x)
            x_feats.append(x)

        # 方便后面对应
        x1, x2, x3, x4 = x_feats

        # ---------- 2. 双向trans layers ----------
        # down: 从x1开始一路down到x4
        y_down = [None] * len(self.translayersdown)
        y_down[0] = self.translayersdown[0](x1)
        for i in range(1, len(self.translayersdown)):
            y_down[i] = self.translayersdown[i](y_down[i - 1])
        y1down, y2down, y3down, y4down = y_down

        # up: 从skip_down开始一路up到y1up
        skip_down = self.skip_down(x0)
        y_up = [None] * len(self.translayersup)
        y_up[0] = self.translayersup[0](skip_down)
        for i in range(1, len(self.translayersup)):
            y_up[i] = self.translayersup[i](y_up[i - 1])  # y3up, y2up, y1up
        y4up, y3up, y2up, y1up = y_up

        # ---------- 3. 多源融合 ----------
        # 逐层 cat + fusion，（深->浅）
        x4 = self.fusion[3](torch.cat([x4, y4down, y4up], dim=1))
        x3 = self.fusion[2](torch.cat([x3, y3down, y3up], dim=1))
        x2 = self.fusion[1](torch.cat([x2, y2down, y2up], dim=1))
        x1 = self.fusion[0](torch.cat([x1, y1down, y1up], dim=1))

        # ---------- 4. decoder ----------
        out3 = self.up3(x4, x3)  # torch.Size([1, 96, 16, 16, 16])
        out2 = self.up2(out3, x2)  # torch.Size([1, 48, 32, 32, 32])
        out1 = self.up1(out2, x1)  # torch.Size([1, 24, 32, 64, 64])

        # ---------- 5. head ----------
        return self.output(skip_in, torch.cat([out1, x0], dim=1))


if __name__ == "__main__":
    device = torch.device("cpu")
    x = torch.randn(1, 1, 32, 128, 128).to(device)
    label = torch.randn(1, 3, 32, 128, 128).to(device)
    model = PRODNet(
        dim_in=1,
        num_classes=3,
        depths=[2, 2, 8, 3],
        embed_dims=[24, 48, 96, 192],
        drop=0.1,
    ).to(device)
    # model = torch.compile(model)
    # with torch.no_grad():
    #     for i in range(10):
    #         y = model(x)
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    #     record_shapes=True,
    #     with_stack=True,
    # ) as prof:
    #     y = model(x)
    y = model(x)
    print(y.shape)
    loss = torch.nn.functional.cross_entropy(y, label)
    loss.backward()
    for name, paras in model.named_parameters():
        if paras.grad is None and paras.requires_grad:
            print(name)
