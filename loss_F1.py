import sys, os, cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
criterion_triplet = nn.TripletMarginLoss(margin=0.1)

def inter_intra_corration_loss(category_corration, margin=0.1):
    N = category_corration[0].size()[0]
    image_specific_prototype = category_corration[0].view(N, 64, -1)   # N, 64, W, H -->> N, 64, WH

    G1_image_specific_prototype = image_specific_prototype[:N // 3, :, :]
    G2_image_specific_prototype = image_specific_prototype[N // 3:2*N // 3, :, :]
    G3_image_specific_prototype = image_specific_prototype[2*N // 3:, :, :]    # N // 3, 64, WH

    G1_consensus_prototype = category_corration[1][0].view(N // 3, 64, -1)  # N // 3, 64, W, H -->> N // 3, 64, WH
    G2_consensus_prototype = category_corration[1][1].view(N // 3, 64, -1)
    G3_consensus_prototype = category_corration[1][2].view(N // 3, 64, -1)

    corration_loss1 = criterion_triplet(G1_consensus_prototype, G1_image_specific_prototype,
                                        G2_consensus_prototype) + criterion_triplet(G1_image_specific_prototype,
                                                                                    G1_consensus_prototype,
                                                                                    G3_consensus_prototype)
    corration_loss2 = criterion_triplet(G2_consensus_prototype, G2_image_specific_prototype,
                                        G3_consensus_prototype) + criterion_triplet(G2_image_specific_prototype,
                                                                                    G2_consensus_prototype,
                                                                                    G1_consensus_prototype)
    corration_loss3 = criterion_triplet(G3_consensus_prototype, G3_image_specific_prototype,
                                        G2_consensus_prototype) + criterion_triplet(G3_image_specific_prototype,
                                                                                    G3_consensus_prototype,
                                                                                    G1_consensus_prototype)
    return corration_loss1 + corration_loss2 + corration_loss3

class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        # return IoU/b
        return IoU

# def IoU_loss(preds, gt):
#
#     N, C, H, W = preds.shape
#     min_tensor = torch.where(preds < gt, preds, gt)    # shape=[N, C, H, W]
#     max_tensor = torch.where(preds > gt, preds, gt)    # shape=[N, C, H, W]
#     min_sum = min_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
#     max_sum = max_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
#     loss = 1 - (min_sum / max_sum).mean()
#
#     return loss

# def _iou(pred, target):
#     b = pred.shape[0]
#     IoU = 0.0
#     for i in range(0, b):
#         # compute the IoU of the foreground
#         Iand1 = torch.sum(target[i, :, :] * pred[i, :, :])
#         Ior1 = torch.sum(target[i, :, :]) + torch.sum(pred[i, :, :]) - Iand1
#         IoU1 = Iand1 / Ior1
#
#         # IoU loss is (1-IoU1)
#         IoU = IoU + (1 - IoU1)
#
#     return IoU / b
#
# class IOU(torch.nn.Module):
#     def __init__(self):
#         super(IOU, self).__init__()
#
#     def forward(self, pred, target):
#         return _iou(pred, target)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    count = (weit > 1).sum().item()
    sum_of_values = torch.sum(weit[weit > 1])
    foreground_weight = sum_of_values // count
    weit = torch.where(mask == 1, foreground_weight * weit, weit)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def iou_loss(pred, mask):
    # pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def floss(prediction, target, beta=0.3, log_like=False):
    prediction = torch.sigmoid(prediction)
    EPS = 1e-10
    N = N = prediction.size(0)
    TP = (prediction * target).view(N, -1).sum(dim=1)
    H = beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
    fmeasure = (1 + beta) * TP / (H + EPS)
    if log_like:
        floss = -torch.log(fmeasure)
    else:
        floss  = (1 - fmeasure)
    floss = floss.mean()
    return floss

def eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    # thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    thlist = torch.linspace(0, 1, num).cuda()

    for i in range(num-1):
        y_temp = torch.logical_and(y_pred>= thlist[i], y_pred < thlist[i+1]).float()
        #y_temp = (y_pred >= thlist[i] ).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def pr_loss(pred, gt):
    # pred = torch.sigmoid(pred)
    prec, recall = eval_pr(pred, gt, 255)
    prec_loss = 1.0 - prec
    recall_loss = 1.0 - recall
    prec_loss = prec_loss.mean()
    recall_loss = recall_loss.mean()

    loss = prec_loss + recall_loss

    return  loss

def eval_pr_original(y_pred, y, num):
    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    # thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    thlist = torch.linspace(0, 1, num).cuda()

    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def pr_loss1(pred, gt):
    beta2 = 0.3
    avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0
    pred = torch.sigmoid(pred)
    for i in range(pred.shape[0]):
        prec, recall = eval_pr(pred[i,0,:], gt[i,0,:], 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        avg_f += f_score
        avg_p += prec
        avg_r += recall
        img_num += 1.0
    avg_p = avg_p / img_num
    avg_r = avg_r / img_num
    Fm = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)

    p = 1 - avg_p
    r = 1 - avg_r
    Fm = 1 - Fm

    loss = p.mean() + r.mean() + Fm.mean()

    return  loss

def object(pred, gt) -> float:
    """
    Calculate the object score.
    """
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    u = torch.mean(gt)
    object_score = u * s_object(fg, gt) + (1 - u) * s_object(bg, 1 - gt)
    return object_score

_EPS = np.spacing(1)
def s_object(pred, gt) -> float:
    x = torch.mean(pred[gt == 1])
    sigma_x = torch.std(pred[gt == 1], unbiased=True)
    score = 2 * x / (torch.pow(x, 2) + 1 + sigma_x + _EPS)
    return score

def region( pred, gt) -> float:
    """
    Calculate the region score.
    """
    x, y = centroid(gt)
    part_info = divide_with_xy(pred, gt, x, y)
    w1, w2, w3, w4 = part_info["weight"]
    # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

    pred1, pred2, pred3, pred4 = part_info["pred"]
    gt1, gt2, gt3, gt4 = part_info["gt"]
    score1 = ssim(pred1, gt1)
    score2 = ssim(pred2, gt2)
    score3 = ssim(pred3, gt3)
    score4 = ssim(pred4, gt4)

    return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

def centroid(matrix) -> tuple:
    """
    To ensure consistency with the matlab code, one is added to the centroid coordinate,
    so there is no need to use the redundant addition operation when dividing the region later,
    because the sequence generated by ``1:X`` in matlab will contain ``X``.

    :param matrix: a data array
    :return: the centroid coordinate
    """
    h, w = matrix.shape
    if matrix.sum() == 0:
        x = torch.round(w / 2)
        y = torch.round(h / 2)
    else:
        area_object = torch.sum(matrix)
        row_ids = torch.arange(h).cuda()
        col_ids = torch.arange(w).cuda()
        x = torch.round(torch.sum(torch.sum(matrix, axis=0) * col_ids) / area_object)
        y = torch.round(torch.sum(torch.sum(matrix, axis=1) * row_ids) / area_object)
    return int(x) + 1, int(y) + 1

def divide_with_xy(pred, gt, x: int, y: int) -> dict:
    """
    Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
    """
    h, w = gt.shape
    area = h * w

    gt_LT = gt[0:y, 0:x]
    gt_RT = gt[0:y, x:w]
    gt_LB = gt[y:h, 0:x]
    gt_RB = gt[y:h, x:w]

    pred_LT = pred[0:y, 0:x]
    pred_RT = pred[0:y, x:w]
    pred_LB = pred[y:h, 0:x]
    pred_RB = pred[y:h, x:w]

    w1 = x * y / area
    w2 = y * (w - x) / area
    w3 = (h - y) * x / area
    w4 = 1 - w1 - w2 - w3

    return dict(
        gt=(gt_LT, gt_RT, gt_LB, gt_RB),
        pred=(pred_LT, pred_RT, pred_LB, pred_RB),
        weight=(w1, w2, w3, w4),
    )

def ssim(pred, gt) -> float:
    """
    Calculate the ssim score.
    """
    h, w = pred.shape
    N = h * w

    x = torch.mean(pred)
    y = torch.mean(gt)

    sigma_x = torch.sum((pred - x) ** 2) / (N - 1)
    sigma_y = torch.sum((gt - y) ** 2) / (N - 1)
    sigma_xy = torch.sum((pred - x) * (gt - y)) / (N - 1)

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

    if alpha != 0:
        score = alpha / (beta + _EPS)
    elif alpha == 0 and beta == 0:
        score = 1
    else:
        score = 0
    return score


def Sm(pred, gt):
    """
    Calculate the S-measure.

    :return: s-measure
    """
    alpha = 0.5
    y = torch.mean(gt)
    if y == 0:
        sm = 1 - torch.mean(pred)
    elif y == 1:
        sm = torch.mean(pred)
    else:
        sm = alpha * object(pred, gt) + (1 - alpha) * region(pred, gt)
        sm = max(0, sm)
    return sm


def Sm_loss1(pred, mask):
    sm = 0.0
    # pred = torch.sigmoid(pred)
    for i in range(pred.shape[0]):
        sm += Sm( pred=pred[i,0,:], gt=mask[i,0,:])
    sm = sm / pred.shape[0]
    sm_loss = 1.0 - sm
    return sm_loss


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score

def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q

def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        if True:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if True:
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        else:
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
    return X.long(), Y.long()

def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q


def Eval_Smeasure(pred, gt):
    alpha, avg_q, img_num = 0.5, 0.0, 0.0
    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        gt[gt >= 0.5] = 1
        gt[gt < 0.5] = 0
        Q = alpha * _S_object(pred, gt) + (1 - alpha) * _S_region(pred, gt)
        if Q.item() < 0:
            Q = torch.FloatTensor([0.0])
    #q = Q.item()
    return 1.0 - Q

def Sm_loss(pred, mask):
    sm = 0.0
    pred = torch.sigmoid(pred)
    for i in range(pred.shape[0]):
        sm += Eval_Smeasure( pred=pred[i,0,:], gt=mask[i,0,:])
    sm = sm / pred.shape[0]
    sm_loss = 1.0 - sm
    return sm_loss