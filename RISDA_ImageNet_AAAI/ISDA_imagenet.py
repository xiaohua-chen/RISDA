from operator import index
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import random

# np.random.seed(42)
# random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
       
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)




class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, beta,index_tail,sth):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        CV_temp = cv_matrix[labels]

        sigma2 = torch.zeros(N, C).cuda()

        for i in range(N):
            W_kj = torch.gather(weight_m, 0, labels[i].view(1, 1).expand(C, A))
            sigma2[i] = beta * (weight_m - W_kj).pow(2).mul(CV_temp[i].view(1, A).expand(C, A)).sum(1)
        
        #---------transfer cv_matrix=============··
        aug_result = y + 0.5 * sigma2
        del sigma2
        #---------transfer mu=============··
        # out_new=sth.get_value()
        # out_new=out_new.cpu()
        # feature_mean=sth.get_feature_mean()
        # feature_mean=feature_mean.cpu()
        # mu=torch.zeros(N,A).cpu()
        # for i in range(N):
        #     if labels[i] in index_tail:
        #         mu=out_new[labels[i]]-feature_mean[labels[i]]

        # dataMean_NxA=mu
        # dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0).cpu()
        # del CV_temp,mu,feature_mean,out_new

        # dataW_NxCxA = NxW_ij - NxW_kj
        # dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        # datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C).cpu()

         #---------transfer CoVariance and mu=============··
        # aug_result = y +  0.5*sigma2+beta* datW_x_detaMean_NxC
        return aug_result

    def forward(self, model, x, target_x, alpha, weights, kg_sigma, index_tail, beta,sth,args):
        y, features = model(x)
       
        self.estimator.update_CV(features.detach(), target_x)

        # transfer the CoVariance
        cv_var=self.get_cv()
        cv_matrix_temp= cv_var.cuda(args.gpu) 
        cv_var_new = torch.matmul(kg_sigma[index_tail],cv_matrix_temp).cuda(args.gpu)
        cv_var[index_tail]=cv_var_new
       
        # 
        self.estimator.CoVariance = cv_var
        isda_aug_y = self.isda_aug(model.module.fc, features, y, target_x, self.estimator.CoVariance.detach(), beta,index_tail,sth)

        loss = F.cross_entropy(isda_aug_y, target_x, weight=weights)
        
        return loss, y

    def get_cv(self):
        return self.estimator.CoVariance