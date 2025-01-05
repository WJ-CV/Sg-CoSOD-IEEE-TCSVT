from collections import OrderedDict
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import pvt_v2
from prototype import G_prototype
from options import args
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU()
    )

def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx

def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy

class Edge_att(nn.Module):
    def __init__(self, in_1, in_2):
        super(Edge_att, self).__init__()
        self.conv_globalinfo = convblock(in_2, in_1, 3, 1, 1)
        self.conv_1 = convblock(in_1, in_1, 1, 1, 0)
        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_1, in_1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.ca = CA(in_1)
        self.sig = nn.Sigmoid()
        self.conv_out = convblock(2*in_1, in_1, 3, 1, 1)
        self.conv_r = nn.Conv2d(in_1, 1, 3, 1, 1)
        self.conv_t = nn.Conv2d(in_1, 1, 3, 1, 1)

    def forward(self, rgb, global_info):
        rgb = self.ca(rgb)
        fus_detail = self.conv_1(rgb + torch.mul(rgb, self.sig(gradient_x(self.conv_r(rgb))+gradient_y(self.conv_r(rgb)))))

        global_info = self.conv_globalinfo(F.interpolate(global_info, rgb.size()[2:], mode='bilinear', align_corners=True))
        fus_semantic = torch.add(rgb, torch.mul(rgb, self.rt_fus(global_info)))

        fus_out = self.conv_out(torch.cat((fus_detail+fus_semantic, fus_detail*fus_semantic), 1))
        return fus_out

class EnLayer(nn.Module):
    def __init__(self, in_channel=64, mid_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x

class LatLayer(nn.Module):
    def __init__(self, in_channel, mid_channel=64):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x

class prediction(nn.Module):
    def __init__(self, in_c_list=[64, 128, 320, 512]):
        super(prediction, self).__init__()
        self.ISP_layer = convblock(in_c_list[-1], in_c_list[0], 3, 1, 1)
        CA_layers = []
        for idx in range(4):
            CA_layers.append(CA(in_c_list[idx]))
        self.CA_layers = nn.ModuleList(CA_layers)

        lat_layers = []
        for idx in range(4):
            lat_layers.append(LatLayer(in_channel=in_c_list[idx], mid_channel=64))
        self.lat_layers = nn.ModuleList(lat_layers)

        dec_layers = []
        for idx in range(4):
            dec_layers.append(EnLayer(in_channel=64, mid_channel=64))
        self.dec_layers = nn.ModuleList(dec_layers)

        self.sig = nn.Sigmoid()
        self.edge_1 = Edge_att(64, 64)
        self.conv_pre = convblock(64 * 2, 64, 3, 1, 1)

    def forward(self, rgb_f, image_specific_prototype):
        for idx in range(4):
            rgb_f[idx] = self.CA_layers[idx](rgb_f[idx])

        feat_ISP = self.ISP_layer(image_specific_prototype)

        feat_down = []
        for idx in range(4):
            p = self.lat_layers[idx](rgb_f[idx])
            feat_down.append(self.upsample_add_sig(p, feat_ISP))

        up_3 = self.dec_layers[3](feat_down[3])
        up_2 = self.dec_layers[2](self.upsample_add(feat_down[2], up_3))
        up_1 = self.dec_layers[1](self.upsample_add(feat_down[1], up_2))
        up_0 = self.dec_layers[0](self.upsample_add(feat_down[0], up_1))

        feat_rgb = [up_0, up_1, up_2, up_3]
        init_pre_f_0 = up_0
        init_pre_f_1 = self.edge_1(feat_down[0], up_0)
        initial_pre_feature = self.conv_pre(torch.cat((init_pre_f_0+init_pre_f_1, init_pre_f_0*init_pre_f_1), 1))

        return feat_rgb, initial_pre_feature

    def upsample_add_sig(self, x, y):
        up_y = F.interpolate(y, x.size()[2:], mode='bilinear')
        return x + x * self.sig(up_y)

    def upsample_add(self, x, y):
        up_y = F.interpolate(y, x.size()[2:], mode='bilinear')
        return x + up_y

class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class C2M(nn.Module):
    def __init__(self, in_2, in_3, in_4):
        super(C2M, self).__init__()

        self.conv_r4_q = convblock(in_4, in_3, 3, 1, 1)
        self.conv_r4_k = convblock(in_4, in_3, 3, 1, 1)

        self.conv_n3 = nn.Conv2d(in_3, in_3, 3, 1, 1)
        self.conv_n2 = nn.Conv2d(in_2, in_2, 3, 1, 1)

        self.conv1d_3_q = nn.Conv1d(in_3, in_2, 3, 1, int((3 - 1) / 2))
        self.conv1d_3_k = nn.Conv1d(in_3, in_2, 3, 1, int((3 - 1) / 2))
        self.conv1d_2_q = nn.Conv1d(in_2, in_2, 5, 1, int((3 - 1) / 2))
        self.conv1d_2_k = nn.Conv1d(in_2, in_2, 5, 1, int((3 - 1) / 2))
        self.conv_2_r = convblock(in_2, in_2, 3, 1, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.gam1 = nn.Parameter(torch.zeros(1))
    def forward(self, x2, x3, x4):
        b, c2, h2, w2 = x2.size()
        b, c3, h3, w3 = x3.size()
        b, c4, h4, w4 = x4.size()

        r_4_q = self.conv_r4_q(x4).view(b, -1, h4 * w4)  # b, c3, l4
        r_4_k = self.conv_r4_k(x4).view(b, -1, h4 * w4)  # b, c3, l4
        r_4_t = r_4_q.permute(0, 2, 1)  # b, l4, c3
        r_3 = self.conv_n3(x3).view(b, -1, h3 * w3)  # b, c3, l3

        r_4_3 = torch.matmul(r_4_t, r_3)  # b, l4, l3
        att_r_4_3 = self.softmax(r_4_3)
        r_3_4 = torch.matmul(r_4_k, att_r_4_3)  # b, c3, l3
        r_3_in_q = self.conv1d_3_q(r_3_4 + r_3)  # b, c2, l3    **********
        r_3_in_k = self.conv1d_3_k(r_3_4 + r_3)  # b, c2, l3    **********

        r_3_in_t = r_3_in_q.permute(0, 2, 1)  # b, l3, c2
        r_2 = self.conv_n2(x2).view(b, -1, h2 * w2)  # b, c2, l2

        r_3_2 = torch.matmul(r_3_in_t, r_2)  # b, l3, l2
        att_r_3_2 = self.softmax(r_3_2)
        r_2_3 = torch.matmul(r_3_in_k, att_r_3_2)  # b, c2, l2
        r_2_in_q = self.conv1d_2_q(r_2_3 + r_2)  # b, c2, l2     **********
        r_2_in_k = self.conv1d_2_k(r_2_3 + r_2)  # b, c2, l2     **********

        r_2_in_t = r_2_in_q.permute(0, 2, 1) # b, l2, c2
        r_2_2 = torch.matmul(r_2_in_t, r_2)  # b, l2, l2
        att_r_2_2 = self.softmax(r_2_2)
        r_2_ = torch.matmul(r_2_in_k, att_r_2_2)

        r_2_out = self.conv_2_r(r_2_.view(b, -1, h2, w2))  # b, c1, h1, w1
        return r_2_out + x2

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.FC2M = C2M(512, 320, 128)
        self.FC2M_1 = C2M(64, 64, 64)
        self.Image_prototype = G_prototype(512, 45, (9, 9))
        self.Image_prototype_1 = G_prototype(64, 45, (9, 9))

        self.G_ISP = convblock(512, 64, 3, 1, 1)

        self.pre_initial = prediction()
        self.pre_final = prediction(in_c_list=[64, 64, 64, 64])

        self.conv_pef1 = convblock(64, 64, 3, 1, 1)
        self.conv_pef2 = convblock(64, 64, 3, 1, 1)
        self.conv_pef3 = convblock(64, 64, 3, 1, 1)

        self.sig = nn.Sigmoid()

    def forward(self, rgb_f, train = False):
        N = rgb_f[0].size()[0]
        # hf_size = rgb_f[3].size()[2:]
        self.Group_prototype = G_prototype(N // 2 * 64, 35, (9, 9)).to('cuda')
        self.Group_prototype_1 = G_prototype(N // 2 * 64, 35, (9, 9)).to('cuda')
        if train:
            #*********initial************
            # category
            global_feature = self.FC2M(rgb_f[3], rgb_f[2], rgb_f[1])   # N, 512, W, H
            image_specific_prototype = self.Image_prototype(global_feature)   # N, 512, W, H

            G_ISP = self.G_ISP(image_specific_prototype)    # N, 64, W, H

            G_ISP_list = [G_ISP[:N//2, :, :, :], G_ISP[N//2:, :, :, :]]
            G_consensus_prototype = []
            for i in range(2):
                trans_i = G_ISP_list[i].view(-1, G_ISP_list[0].size()[2], G_ISP_list[0].size()[3]).unsqueeze(0) #1, N//3 * 64, W, H
                G_con_prototype = self.Group_prototype(trans_i)
                G_consensus_prototype_ = G_con_prototype.squeeze(0).view(N//2, 64, G_ISP_list[0].size()[2], G_ISP_list[0].size()[3])
                G_consensus_prototype.append(G_consensus_prototype_) #N//3, 64, W, H

            initial_category_corration = [G_ISP, G_consensus_prototype]

            # saliency
            feat_rgb, initial_prediction_feature = self.pre_initial(rgb_f, image_specific_prototype)

            #**************Refine**************
            # category
            PreEF1 = feat_rgb[1] + feat_rgb[1] * self.conv_pef1(F.interpolate(initial_prediction_feature, feat_rgb[1].size()[2:], mode='bilinear',
                                                       align_corners=True))
            PreEF2 = feat_rgb[2] + feat_rgb[2] * self.conv_pef2(F.interpolate(initial_prediction_feature, feat_rgb[2].size()[2:], mode='bilinear',
                                                       align_corners=True))
            PreEF3 = feat_rgb[3] + feat_rgb[3] * self.conv_pef3(F.interpolate(initial_prediction_feature, feat_rgb[3].size()[2:], mode='bilinear',
                                                       align_corners=True))
            Refine_global_feature = self.FC2M_1(PreEF3, PreEF2, PreEF1)
            R_image_specific_prototype = self.Image_prototype_1(Refine_global_feature)   # N, 512, W, H

            R_G_ISP = R_image_specific_prototype    # N, 64, W, H

            R_G_ISP_list = [R_G_ISP[:N//2, :, :, :], R_G_ISP[N//2:, :, :, :]]
            R_G_consensus_prototype = []
            for i in range(2):
                trans_i = R_G_ISP_list[i].view(-1, R_G_ISP_list[0].size()[2], R_G_ISP_list[0].size()[3]).unsqueeze(0) #1, N//3 * 64, W, H
                R_G_con_prototype = self.Group_prototype_1(trans_i)
                R_G_consensus_prototype_ = R_G_con_prototype.squeeze(0).view(N//2, 64, R_G_ISP_list[0].size()[2], R_G_ISP_list[0].size()[3])
                R_G_consensus_prototype.append(R_G_consensus_prototype_) #N//3, 64, W, H

            final_category_corration = [R_G_ISP, R_G_consensus_prototype]

            # saliency
            R_feat_rgb, final_prediction_feature = self.pre_final(feat_rgb, R_image_specific_prototype)   # N, 64, 8*W, 8*H

            prediction_feature = [final_prediction_feature, initial_prediction_feature]

            return prediction_feature, final_category_corration, initial_category_corration
        else:
            #*********initial************
            # category
            global_feature = self.FC2M(rgb_f[3], rgb_f[2], rgb_f[1])   # N, 512, W, H
            image_specific_prototype = self.Image_prototype(global_feature)   # N, 512, W, H

            # saliency
            feat_rgb, initial_prediction_feature = self.pre_initial(rgb_f, image_specific_prototype)

            #**************Refine**************
            # category
            PreEF1 = feat_rgb[1] + feat_rgb[1] * self.conv_pef1(F.interpolate(initial_prediction_feature, feat_rgb[1].size()[2:], mode='bilinear',
                                                       align_corners=True))
            PreEF2 = feat_rgb[2] + feat_rgb[2] * self.conv_pef2(F.interpolate(initial_prediction_feature, feat_rgb[2].size()[2:], mode='bilinear',
                                                       align_corners=True))
            PreEF3 = feat_rgb[3] + feat_rgb[3] * self.conv_pef3(F.interpolate(initial_prediction_feature, feat_rgb[3].size()[2:], mode='bilinear',
                                                       align_corners=True))
            Refine_global_feature = self.FC2M_1(PreEF3, PreEF2, PreEF1)
            R_image_specific_prototype = self.Image_prototype_1(Refine_global_feature)   # N, 512, W, H

            # saliency                                         align_corners=True))
            R_feat_rgb, final_prediction_feature = self.pre_final(feat_rgb, R_image_specific_prototype)   # N, 64, 8*W, 8*H

            prediction_feature = [final_prediction_feature, initial_prediction_feature]

            return prediction_feature

class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(pvt_v2, backbone)()
        if pretrained:
            checkpoint = torch.load('../pvt_v2_b3.pth', map_location='cpu')
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.encoder.load_state_dict(checkpoint_model, strict=False)

def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model

class CoSODnet(nn.Module):
    def __init__(self):
        super(CoSODnet, self).__init__()
        model = Encoder()
        self.rgb_net = model.encoder
        self.decoder = Decoder()
        self.conv_input = nn.Conv2d(4 * 3, 3, 3, 1, 1)

        self.score_final = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_initial = nn.Conv2d(64, 1, 1, 1, 0)

        self.score = nn.Conv2d(64 * 4, 1, 1, 1, 0)

        self.sig = nn.Sigmoid()

    def forward(self, rgb, depth, is_train = False):
        depth = torch.cat((depth, depth, depth), 1)
        input = self.conv_input(torch.cat((rgb, depth, rgb + depth, rgb * depth), 1))
        rgb_f = self.rgb_net(input)
        if is_train:
            prediction_feature, final_category_corration, initial_category_corration = self.decoder(rgb_f, train = is_train)

            final_pre, initial_pre = prediction_feature[0], prediction_feature[1]
            score_final = self.score_final(F.interpolate(final_pre, (args.size, args.size), mode='bilinear', align_corners=True))
            score_initial = self.score_initial(F.interpolate(initial_pre, (args.size, args.size), mode='bilinear', align_corners=True))

            score_pre = torch.cat((final_pre + torch.mul(final_pre, self.sig(initial_pre)),
                                          initial_pre + torch.mul(initial_pre, self.sig(final_pre)), final_pre + initial_pre, final_pre * initial_pre), 1)
            score = self.score(F.interpolate(score_pre, (args.size, args.size), mode='bilinear', align_corners=True))

            score_prediction = [self.sig(score), self.sig(score_final), self.sig(score_initial)]
            return score_prediction, final_category_corration, initial_category_corration
        else:
            prediction_feature = self.decoder(rgb_f, train = is_train)

            final_pre, initial_pre = prediction_feature[0], prediction_feature[1]
            score_final = self.score_final(F.interpolate(final_pre, (args.size, args.size), mode='bilinear', align_corners=True))
            score_initial = self.score_initial(F.interpolate(initial_pre, (args.size, args.size), mode='bilinear', align_corners=True))

            score_pre = torch.cat((final_pre + torch.mul(final_pre, self.sig(initial_pre)),
                                          initial_pre + torch.mul(initial_pre, self.sig(final_pre)), final_pre + initial_pre, final_pre * initial_pre), 1)
            score = self.score(F.interpolate(score_pre, (args.size, args.size), mode='bilinear', align_corners=True))

            score_prediction = [self.sig(score), self.sig(score_final), self.sig(score_initial)]

            return score_prediction