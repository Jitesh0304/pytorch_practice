"""
PointPillars has exactly 3 modules that work in sequence. Data flows from left to right — the output of each module becomes 
input to the next.

1) PillarFeatureNet
A small neural network that learns a 64-dim feature for each pillar. Uses a simplified PointNet internally.

2) Backbone (SECOND CNN)
Scatters pillar features into a 2D BEV grid, then runs a fast CNN with multi-scale feature pyramids.

3) Detection Head
For every anchor on the BEV grid, predicts: Does an object exist here? If yes, what class + box offsets?

"""


import torch
import torch.nn as nn

class PillarFeatureNet(nn.Module):
    """
    Input : (B, P, 32, 9)  — batch, pillars, points, features
    Output: (B, P, 64)     — one 64-dim vector per pillar
    """
    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()

        # Simple shared MLP (same weights applied to every point)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, pillars):
        # pillars: (B, P, 32, 9)
        B, P, N, C = pillars.shape

        # Flatten to apply linear layer to every point
        x = pillars.view(B * P * N, C)    # (B*P*32, 9)
        x = self.mlp(x)                     # (B*P*32, 64)
        x = x.view(B, P, N, 64)           # (B, P, 32, 64)

        # Max-pool over points → one vector per pillar
        x, _ = x.max(dim=2)              # (B, P, 64)
        return x



class PillarScatter(nn.Module):
    """
    Places pillar features back into their (x,y) grid positions.
    Input : pillar_feats (B, P, 64), coords (B, P, 2), grid size
    Output: pseudo-image  (B, 64, H, W)
    """
    def __init__(self, grid_h, grid_w, num_features=64):
        super().__init__()
        self.H = grid_h
        self.W = grid_w
        self.C = num_features

    def forward(self, pillar_feats, coords, num_pillars):
        B = pillar_feats.shape[0]
        bev = pillar_feats.new_zeros(B, self.C, self.H, self.W)

        for b in range(B):
            n  = num_pillars[b]
            gx = coords[b, :n, 0]  # grid x indices
            gy = coords[b, :n, 1]  # grid y indices
            bev[b, :, gy, gx] = pillar_feats[b, :n].T  # scatter

        return bev  # (B, 64, H, W)


class Backbone(nn.Module):
    """
    A simple CNN that processes the BEV pseudo-image.
    Input : (B, 64, H, W)
    Output: (B, 384, H/8, W/8)  — multi-scale features
    """
    def _block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def __init__(self):
        super().__init__()
        # Three downsampling blocks (stride=2 halves spatial size)
        self.block1 = nn.Sequential(self._block(64,  64, stride=2),
                                      self._block(64,  64))
        self.block2 = nn.Sequential(self._block(64,  128, stride=2),
                                      self._block(128, 128))
        self.block3 = nn.Sequential(self._block(128, 256, stride=2),
                                      self._block(256, 256))

        # Upsample all blocks to same resolution and concatenate
        self.up1 = nn.ConvTranspose2d(64,  128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=4)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        # FPN-style: upsample and concatenate → richer features
        u1 = self.up1(x1)
        u2 = self.up2(x2)
        u3 = self.up3(x3)
        return torch.cat([u1, u2, u3], dim=1)  # (B, 384, H/2, W/2)



class DetectionHead(nn.Module):
    """
    For each spatial location on BEV map, predicts:
      - cls_logits : (B, num_anchors * num_classes, H, W)
      - box_preds  : (B, num_anchors * 7,           H, W)
    """
    def __init__(self, in_channels=384, num_anchors=2, num_classes=3):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, num_anchors * 7, 1)
        self.dir_head = nn.Conv2d(in_channels, num_anchors * 2, 1)  # direction

    def forward(self, x):
        return {
            "cls": self.cls_head(x),
            "box": self.box_head(x),
            "dir": self.dir_head(x),
        }


class PointPillars(nn.Module):
    """Complete PointPillars model"""
    def __init__(self, num_classes=3,
                 grid_h=625, grid_w=625,  # 100m / 0.16m ≈ 625
                 num_anchors=2):
        super().__init__()
        self.pfn      = PillarFeatureNet(in_channels=9, out_channels=64)
        self.scatter  = PillarScatter(grid_h, grid_w, num_features=64)
        self.backbone = Backbone()
        self.head     = DetectionHead(in_channels=384,
                                        num_anchors=num_anchors,
                                        num_classes=num_classes)

    def forward(self, pillars, coords, num_pillars):
        # pillars: (B, P, 32, 9)
        feats = self.pfn(pillars)               # → (B, P, 64)
        bev   = self.scatter(feats, coords, num_pillars)  # → (B, 64, H, W)
        bev   = self.backbone(bev)              # → (B, 384, H/2, W/2)
        preds = self.head(bev)                  # → dict of cls/box/dir
        return preds




