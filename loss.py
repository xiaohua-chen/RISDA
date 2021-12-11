import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(var_temp.permute(1, 2, 0), var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)).detach() + additional_CV.detach()
        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)


class RISDA_CE(nn.Module):
    def __init__(self, feature_num, class_num, cls_num_list, max_m=0.5, s=30):
        super(RISDA_CE, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s

    def RISDA(self, fc_kg_new, features, y_s, labels_s, s_cv_matrix, alpha,kg,out_new,feature_mean,beta):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m=fc_kg_new
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        s_CV_temp = s_cv_matrix[labels_s]
        #use beta calculate sigma_ij
        sigma2 = beta * torch.bmm(torch.bmm(NxW_ij - NxW_kj, s_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        #reasnoning mu in loss function
        dataMean_NxA=out_new[labels_s]
     
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0).cuda()
        del s_CV_temp
        dataW_NxCxA = NxW_ij - NxW_kj
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)
        
        aug_result = y_s +  0.5*sigma2+ alpha*datW_x_detaMean_NxC
        return aug_result

    def forward(self, fc, features, y_s, labels, alpha, weights, cv, manner,kg,out_new,feature_mean,beta,head):

        self.estimator.update_CV(features.detach(), labels)
        # reasoning covariance  head=20
        tail=self.class_num -head
        cv_var=self.get_cv()
        cv_matrix_temp= cv_var.view(cv_var.size(0), -1).cuda() 
        kg=kg.cuda()
        
        cv_var_new = torch.matmul(kg[head:],cv_matrix_temp ).view(tail,64,-1)
        cv_var=cv_var.cuda()
        cv_var_new=cv_var_new.cuda()
        new_cv=torch.cat((cv_var[:head],cv_var_new),0)
        cv=new_cv
        # update covariance
        self.estimator.CoVariance = new_cv
        
        fc_kg=list(fc.named_leaves())[0][1]
        fc_kg_new=fc_kg

        aug_y = self.RISDA(fc_kg_new, features, y_s, labels, cv, alpha,kg,out_new,feature_mean,beta)
        loss = F.cross_entropy(aug_y, labels, weight=weights)
        return loss

    def get_cv(self):
        return self.estimator.CoVariance

    def update_cv(self, cv):
        self.estimator.CoVariance = cv