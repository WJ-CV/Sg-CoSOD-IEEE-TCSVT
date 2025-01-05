import torch
from torch import nn
from torch.nn import functional as F

class G_prototype(nn.Module):
    def __init__(self, in_c, num_p, size):
        super(G_prototype, self).__init__()
        self.num_cluster = num_p
        self.netup = torch.nn.Sequential(
                torch.nn.Conv2d(in_c, num_p, 3, padding=1)
                )
        self.centroids = torch.nn.Parameter(torch.rand(num_p, in_c))   #(24, 256)       K, C

        self.upfc = torch.nn.Linear(num_p*in_c, in_c)
        self.upfs = torch.nn.Linear(num_p * size[0] * size[1], size[0] * size[1])
        self.transform = torch.nn.Sequential(
            nn.Conv2d(3*in_c, in_c, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.ReLU(inplace=False),
            )

    def UP(self, scene):
        num_cluster = self.num_cluster
        x = scene
        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)    # N, C, H, W       #
        soft_assign = self.netup(x)                #

        soft_assign = F.softmax(soft_assign, dim=1)  # N, K, H, W
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)  # N, K, HW
    
        x_flatten = x.view(N, C, -1)  # N, C, HW

        centroid = self.centroids       #K, C)

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3) #                   N, K, C, HW
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)#         1, K, C, HW

        residual = x1 - x2        # N, K, C, HW
        residual = residual * soft_assign.unsqueeze(2)    # N, K, C, HW   *  N, K, 1, HW  =  N, K, C, HW

        up_c = residual.sum(dim=-1)    # N, K, C
        up_c = F.normalize(up_c, p=2, dim=2)  # N, K, C
        up_c = up_c.view(x.size(0), -1)   # N, KC
        up_c = F.normalize(up_c, p=2, dim=1)   # N, KC
        up_c = self.upfc(up_c).unsqueeze(2).unsqueeze(3).repeat(1,1,W,H) # N, C, W, H

        up_s = residual.sum(dim=2)    # N, K, HW
        up_s = F.normalize(up_s, p=2, dim=2)  # N, K, HW
        up_s = up_s.view(x.size(0), -1)   # N, KHW
        up_s = F.normalize(up_s, p=2, dim=1)   # N, KHW
        up_s = self.upfs(up_s).unsqueeze(1) # N, 1, HW
        up_s = up_s.view(N, 1, W, H).repeat(1, C, 1, 1) # N, C, W, H

        return up_c, up_s, centroid

    def forward(self, feature):

        up_c, up_s, centroid = self.UP(feature)
        new_feature = torch.cat((feature, up_c * feature, up_s * feature), dim=1)
        new_feature = self.transform(new_feature)

        return new_feature

