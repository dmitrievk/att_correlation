import scipy
import numpy as np
from emd import emd


def cal_cc_score(Att_1, Att_2):
    """
    Compute CC score between two attention maps
    """
    eps = 2.2204e-16 #regularization value
    if np.max(Att_1) > 0:
        Map_1 = Att_1-np.mean(Att_1) / np.std(Att_1)
    else:
        Map_1 = Att_1 + eps
    if np.max(Att_2) > 0:
        Map_2 = Att_2-np.mean(Att_2) / np.std(Att_2)
    else:
        Map_2 = Att_2 + eps

    return np.corrcoef(Map_1.reshape(-1), Map_2.reshape(-1))[0][1]

def cal_sim_score(Att_1, Att_2):
    """
    Compute SIM score between two attention maps
    """
    if np.sum(Att_1)>0:
        Map_1 = Att_1/np.sum(Att_1)
    else:
        Map_1=Att_1
    if np.sum(Att_2)>0:
        Map_2 = Att_2/np.sum(Att_2)
    else:
        Map_2=Att_2

    sim_score = np.minimum(Map_1,Map_2)

    return np.sum(sim_score)


def cal_kld_score(Att_1,Att_2): #recommand Att_1 to be free-viewing attention
    """
    Compute KL-Divergence score between two attention maps
    """
    eps = 2.2204e-16 #regularization value
    if np.sum(Att_1)>0:
        Map_1 = Att_1/np.sum(Att_1)
    else:
        Map_1=Att_1
    if np.sum(Att_2)>0:
        Map_2 = Att_2/np.sum(Att_2)
    else:
        Map_2=Att_2

    kl_score = Map_2*np.log(eps+Map_2/(Map_1+eps))
    return np.sum(kl_score)

def cal_emd_score(Att_1,Att_2):
    """
    Compute Earth Mover Distance between two attention maps
    """
    if np.sum(Att_1)>0:
        Map_1 = Att_1/np.sum(Att_1)
    else:
        Map_1=Att_1
    if np.sum(Att_2)>0:
        Map_2 = Att_2/np.sum(Att_2)
    else:
        Map_2=Att_2

    emd_score = emd(Map_1,Map_2)

    return emd_score


def cal_auc_score(salMap, fixMap):
    """
    compute the AUC score for saliency prediction
    """
    fixMap = (fixMap*255>200).astype(int)
    if np.sum(fixMap)==0:
        return 0.5
    salShape = salMap.shape
    fixShape = fixMap.shape

    predicted = salMap.reshape(salShape[0]*salShape[1], -1,
                               order='F').flatten()
    actual = fixMap.reshape(fixShape[0]*fixShape[1], -1,
                            order='F').flatten()
    labelset = np.arange(2)

    auc = area_under_curve(predicted, actual, labelset)
    return auc if not np.isnan(auc) else 0.5

def area_under_curve(predicted, actual, labelset):
    tp, fp = roc_curve(predicted, actual, np.max(labelset))
    auc = auc_from_roc(tp, fp)
    return auc

def auc_from_roc(tp, fp):
    h = np.diff(fp)
    auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
    return auc

def roc_curve(predicted, actual, cls):
    si = np.argsort(-predicted)
    tp = np.cumsum(np.single(actual[si]==cls))
    fp = np.cumsum(np.single(actual[si]!=cls))
    tp = tp/np.sum(actual==cls)
    fp = fp/np.sum(actual!=cls)
    tp = np.hstack((0.0, tp, 1.0))
    fp = np.hstack((0.0, fp, 1.0))
    return tp, fp
