import sys
sys.path.append('./util')
import numpy as np
from salicon.salicon import SALICON
import cv2
from evaluation import cal_cc_score,cal_sim_score,cal_kld_score,cal_emd_score
import os
from scipy.stats import spearmanr

multi_Q = 1

def init_metrics(metrics):
    metric = dict()
    for k in metrics:
        metric[k] = 0
    metric['count'] = 0

    return metric

def main():
    img_rows,img_cols = 300,400
    #initializing salicon data
    sal_anno = '/home/eric/Desktop/experiment/salicon/salicon-api/annotations/fixations_train2014.json'
    salicon = SALICON(sal_anno)

    #loading VQA data
    vqa_dict = np.load('valid_data_train.npy')
    question_bank = np.load('question_type.npy')
    answer_bank = np.load('answer_type.npy')
    vqa_dir = '/media/eric/New Volume/VQA/VQA_HAT/vqahat_train'

    #defining data structure
    metrics=['cc','sim','kld','emd','spearmanr']
    que_score = dict()
    ans_score = dict()
    overall_score = dict()

    for question in question_bank:
        que_score[question] = init_metrics(metrics)
    for answer in answer_bank:
        ans_score[answer] = init_metrics(metrics)
    overall_score = init_metrics(metrics)

    #main loop for comparing different attention maps
    nan_count_q = dict()
    nan_count_a = dict()
    nan_corr_q = dict()
    nan_corr_a = dict()
    nan_count = 0
    nan_corr = 0
    for i in question_bank:
        nan_count_q[i]=0
        nan_corr_q[i]=0
    for i in answer_bank:
        nan_count_a[i]=0
        nan_corr_a[i] = 0

    for cur_data in vqa_dict:
        question_id = cur_data['question_id']
        question_type = cur_data['question_type']
        answer_type = cur_data['answer_type']
        img_id = cur_data['img_id']

        #load vqa attention map
        vqa_img = os.path.join(vqa_dir,str(question_id)+'_1.png')
        que_att_map = cv2.imread(vqa_img)
        que_att_map = que_att_map[:,:,0]
        que_att_map = cv2.resize(que_att_map, (img_cols, img_rows),interpolation = cv2.INTER_LINEAR)
        que_att_map = que_att_map.astype('float32')
        que_att_map /= 255

        #load free-viewing attention map
        annIds = salicon.getAnnIds(img_id)
        anns = salicon.loadAnns(annIds)
        fv_att_map = salicon.showAnns(anns)
        fv_att_map = cv2.resize(fv_att_map, (img_cols, img_rows),interpolation = cv2.INTER_LINEAR)

        #computing scores for different metrics
        cc = cal_cc_score(fv_att_map,que_att_map)
        sim = cal_sim_score(fv_att_map,que_att_map)
        kld = cal_kld_score(fv_att_map,que_att_map)
        emd = cal_emd_score(fv_att_map,que_att_map)
        rank_corr,_ = spearmanr(fv_att_map.reshape(-1),que_att_map.reshape(-1))

        #storing data in a naive way
        if np.isnan(cc):
            cc = 0
            nan_count_q[question_type]+=1
            nan_count_a[answer_type]+=1
            nan_count+=1
        if np.isnan(rank_corr):
            rank_corr=0
            nan_corr_q[question_type]+=1
            nan_corr_a[answer_type]+=1
            nan_corr+=1

        que_score[question_type]['cc']+=cc
        que_score[question_type]['sim']+=sim
        que_score[question_type]['spearmanr']+=rank_corr
        que_score[question_type]['kld']+=kld
        que_score[question_type]['emd']+=emd
        que_score[question_type]['count']+=1

        ans_score[answer_type]['cc']+=cc
        ans_score[answer_type]['sim']+=sim
        ans_score[answer_type]['spearmanr']+=rank_corr
        ans_score[answer_type]['kld']+=kld
        ans_score[answer_type]['emd']+=emd
        ans_score[answer_type]['count']+=1

        overall_score['cc']+=cc
        overall_score['sim']+=sim
        overall_score['spearmanr']+=rank_corr
        overall_score['kld']+=kld
        overall_score['emd']+=emd
        overall_score['count']+=1

    #computing average score
    for q_type in question_bank:
        for cur_metric in metrics:
            if cur_metric=='cc':
                que_score[q_type][cur_metric]/=que_score[q_type]['count']-nan_count_q[q_type]
            elif cur_metric=='spearmanr':
                que_score[q_type][cur_metric]/=que_score[q_type]['count']-nan_corr_q[q_type]
            else:
                que_score[q_type][cur_metric]/=que_score[q_type]['count']

    for a_type in answer_bank:
        for cur_metric in metrics:
            if cur_metric=='cc':
                ans_score[a_type][cur_metric]/=ans_score[a_type]['count']-nan_count_a[a_type]
            elif cur_metric=='spearmanr':
                ans_score[a_type][cur_metric]/=ans_score[a_type]['count']-nan_corr_a[a_type]
            else:
                ans_score[a_type][cur_metric]/=ans_score[a_type]['count']

    for cur_metric in metrics:
        if cur_metric=='cc':
            overall_score[cur_metric]/=overall_score['count']-nan_count
        elif cur_metric=='spearmanr':
            overall_score[cur_metric]/=overall_score['count']-nan_corr
        else:
            overall_score[cur_metric]/=overall_score['count']

    np.save('question_score',que_score)
    np.save('answer_score',ans_score)
    np.save('overall_score',overall_score)

def multi_question():
    img_rows,img_cols = 300,400
    vqa_dir = '/media/eric/New Volume/VQA/VQA_HAT/vqahat_train'
    IQ_pair =  np.load('multi_question.npy')
    metrics=['cc','sim','kld','emd','spearmanr']
    inter_score = dict()
    score = init_metrics(metrics)

    #main loop for comparing different attention maps
    nan_cc = 0
    nan_corr = 0
    for cur in IQ_pair.item():
        sal_map=[]
        for q_id in IQ_pair.item()[cur]:
            I_dir = os.path.join(vqa_dir,str(q_id)+'_1.png')
            I=cv2.imread(I_dir)
            I=cv2.resize(I, (img_cols, img_rows),interpolation = cv2.INTER_LINEAR)
            I=I[:,:,0]
            I = I.astype('float32')
            sal_map.append(I)
        tmp_pair = [(0,1),(0,2),(1,2)] if len(sal_map)==3 else [(0,1)]
        if len(sal_map)==1:
            continue
        tmp_cc = 0
        tmp_kld = 0
        tmp_sim = 0
        tmp_corr = 0
        nan_corr_ = 0
        nan_cc_ = 0
        for pair in tmp_pair:
            cc = cal_cc_score(sal_map[pair[0]],sal_map[pair[1]])
            tmp_kld += cal_kld_score(sal_map[pair[0]],sal_map[pair[1]])
            tmp_sim += cal_sim_score(sal_map[pair[0]],sal_map[pair[1]])
            corr,_ = spearmanr(sal_map[pair[0]].reshape(-1),sal_map[pair[1]].reshape(-1))

            if np.isnan(cc):
                nan_cc_ +=1
            else:
                tmp_cc+=cc
            if np.isnan(corr):
                nan_corr_+=1
            else:
                tmp_corr+=corr
        score['count']+=1
        score['kld']+=tmp_kld/len(sal_map)
        score['sim']+=tmp_sim/len(sal_map)
        if len(sal_map)-nan_cc_>0:
            score['cc']+=tmp_cc/(len(sal_map)-nan_cc_)
        else:
            nan_cc+=1
        if len(sal_map)-nan_corr_>0:
            score['spearmanr']+=tmp_corr/(len(sal_map)-nan_corr_)
        else:
            nan_corr+=1

    for metric in metrics:
        if metric == 'cc':
            score[metric] /= score['count']-nan_cc
        elif metric == 'spearmanr':
            score[metric] /= score['count']-nan_corr
        else:
            score[metric] /= score['count']

    np.save('multi_question_score',score)


if multi_Q==0:
    main()
else:
    multi_question()
