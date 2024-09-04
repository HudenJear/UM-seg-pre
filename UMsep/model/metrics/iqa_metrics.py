from scipy import stats
import numpy as np


def calculate_l1(pred,gt,**kwargs):
    """
    Calculate the L1 loss for 2 imgs.
    Order of input did not effect the result.
    """
    # print(pred,gt)
    # pred=pred.flatten()
    # gt=gt.flatten()
    # print(pred,gt)

    variation=np.abs(pred-gt)
    l1=variation.sum().sum()
    return l1


def calculate_srcc(pred,gt,**kwargs):
    """
    Calculate the Spearman rank CC for the score.
    Order of input did not effect the result.
    """
    srcc,_=stats.spearmanr(pred,gt)
    return srcc

def calculate_plcc(pred,gt,**kwargs):
    """
    Calculate the Pearson linear CC for the score.
    Order of input did not effect the result.
    """
    plcc,_=stats.pearsonr(pred,gt)
    return plcc

def calculate_rmse(pred,gt,**kwargs):
    """
    Calculate the RMSE for the score.
    Order of input did not effect the result.
    """
    mse=np.sum((pred-gt)**2/len(pred))
    rmse=np.sqrt(mse)

    return  rmse

def calculate_PRsum(pred,gt,**kwargs):
    """
    Calculate the RMSE for the score.
    Order of input did not effect the result.
    """
    srcc,_=stats.spearmanr(pred,gt)
    plcc,_=stats.pearsonr(pred,gt)

    return  plcc+srcc
