import scipy
import numpy as np
from emd import emd


def cal_cc_score(Att_1, Att_2):
    """
    Compute CC score between two attention maps
    """
    eps = 2.2204e-16 #regularization value
    Map_1 = Att_1 - np.mean(Att_1)
    if np.max(Map_1) > 0:
        Map_1 = Map_1 / np.std(Map_1)
    Map_2 = Att_2 - np.mean(Att_2)
    if np.max(Map_2) > 0:
        Map_2 = Map_2 / np.std(Map_2)
    if np.sum(Map_1)==0:
        Map_1+=eps
    if np.sum(Map_2)==0:
        Map_2+=eps

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
