import collections
import pickle
import os.path as osp
import numpy as np
import click
from collections import defaultdict
import random
from copy import deepcopy
import time
import pandas as pd
import pdb
import argparse
import logging
import os

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--train_num', type = int, default = 0, help = 'how many train data to generate')
    #parser.add_argument('--validation_num', type=int, default=0, help='how many validation data to generate')
    parser.add_argument('--test_num', type=int, default=0, help='how many test data to generate')
    parser.add_argument('--path', default='data/', help='path to data')
    parser.add_argument('--seed', type = int, default='6', help='randomness of the data')
    #parser.add_argument('--query_name', default=3 , help='Which one do u wanna generate, index of query structure')
    return parser.parse_args()

def load_data(path, dataset):
    test_path = path + dataset + '/test.txt'
    valid_path = path + dataset + '/valid.txt'
    train_path = path + dataset + '/train.txt'
    train_df = pd.read_csv(train_path, sep='	')
    valid_df = pd.read_csv(valid_path, sep='	')
    test_df = pd.read_csv(test_path, sep='	')
    train_df = train_df.set_axis(['e1', 'r', 'e2'], axis=1)
    valid_df = valid_df.set_axis(['e1', 'r', 'e2'], axis=1)
    test_df = test_df.set_axis(['e1', 'r', 'e2'], axis=1)
    return train_df, valid_df, test_df

def create_1p(e1, df):
    selected = df[df['e1'] == e1]
    randid = random.randrange(0, len(selected.index))
    r = selected.iloc[randid]['r'] # choose a random relationship from the data have selected e1
    e2 = selected[selected['r'] == r]['e2'].values # select all e2 that meets the relationship r
    return r, e2

def create_4p(train_df, valid_df, test_df, nums, seed):
    random.seed(seed)
    easy_df = pd.concat([train_df, valid_df])
    all_df = pd.concat([train_df, valid_df, test_df])
    e1_all = []
    r_all = []
    answers_all = []
    answers_easy = []
    answers_hard = []
    i = 0
    while i < nums:
        e = []
        r = np.zeros(4)
        randid = random.randrange(0, len(test_df.index))
        e.append([test_df.iloc[randid]['e1']])
        r[0], e_next = create_1p(e[0][0], test_df)
        e.append(e_next.tolist())
        for round in range(1,4):
            randid = random.randrange(0, len(e[round]))
            e_prev = e[round][randid]
            r[round], _ = create_1p(e_prev, train_df)
            e_next = train_df[(train_df['e1'].isin(e[round])) & (train_df['r'] == r[round])]['e2'].values
            e.append(list(set(e_next)))
        if len(e_next) == 0:
            continue
        e2_all = [e[0]]
        e2_easy = [e[0]]
        for round in range(4):
            easy_next = easy_df[(easy_df['e1'].isin(e2_easy[round])) & (easy_df['r'] == r[round])]['e2'].values
            all_next = all_df[(all_df['e1'].isin(e2_all[round])) & (all_df['r'] == r[round])]['e2'].values
            e2_easy.append(list(set(easy_next)))
            e2_all.append(list(set(all_next)))
        if len([item for item in e2_all[4] if item not in e2_easy[4]]) != 0:
            e1_all.append(e[0][0])
            r_all.append(r.astype(int))
            answers_easy.append(e2_easy[4])
            answers_all.append(e2_all[4])
            answers_hard.append([item for item in e2_all[4] if item not in e2_easy[4]])
            i = i + 1
    queries, easy = convert_format(e1_all, r_all, answers_easy, '4p')
    queries, hard = convert_format(e1_all, r_all, answers_hard, '4p')
    queries, all = convert_format(e1_all, r_all, answers_all, '4p')
    return queries, easy, hard, all

def create_5p(train_df, valid_df, test_df, nums, seed):
    random.seed(seed+1)
    easy_df = pd.concat([train_df, valid_df])
    all_df = pd.concat([train_df, valid_df, test_df])
    e1_all = []
    r_all = []
    answers_all = []
    answers_easy = []
    answers_hard = []
    i = 0
    while i < nums:
        e = []
        r = np.zeros(5)
        randid = random.randrange(0, len(test_df.index))
        e.append([test_df.iloc[randid]['e1']])
        r[0], e_next = create_1p(e[0][0], test_df)
        e.append(e_next.tolist())
        for round in range(1,5):
            randid = random.randrange(0, len(e[round]))
            e_prev = e[round][randid]
            r[round], _ = create_1p(e_prev, train_df)
            e_next = train_df[(train_df['e1'].isin(e[round])) & (train_df['r'] == r[round])]['e2'].values
            e.append(list(set(e_next)))
        if len(e_next) == 0:
            continue
        e2_all = [e[0]]
        e2_easy = [e[0]]
        for round in range(5):
            easy_next = easy_df[(easy_df['e1'].isin(e2_easy[round])) & (easy_df['r'] == r[round])]['e2'].values
            all_next = all_df[(all_df['e1'].isin(e2_all[round])) & (all_df['r'] == r[round])]['e2'].values
            e2_easy.append(list(set(easy_next)))
            e2_all.append(list(set(all_next)))
        if len([item for item in e2_all[5] if item not in e2_easy[5]]) != 0:
            e1_all.append(e[0][0])
            r_all.append(r.astype(int))
            answers_hard.append([item for item in e2_all[5] if item not in e2_easy[5]])
            answers_easy.append(e2_easy[5])
            answers_all.append(e2_all[5])
            i = i + 1
    queries, easy = convert_format(e1_all, r_all, answers_easy, '5p')
    queries, hard = convert_format(e1_all, r_all, answers_hard, '5p')
    queries, all = convert_format(e1_all, r_all, answers_all, '5p')
    return queries, easy, hard, all

def create_4i(train_df, valid_df, test_df, nums, seed):
    random.seed(seed+2)
    easy_df = pd.concat([train_df, valid_df])
    all_df = pd.concat([train_df, valid_df, test_df])
    e1_all = []
    r_all = []
    answers_all = []
    answers_easy = []
    answers_hard = []
    i = 0
    while i < nums:
        randid = random.randrange(0, len(test_df.index))
        e2 = test_df.iloc[randid]['e2']
        r11 = test_df.iloc[randid]['r']
        e11 = test_df.iloc[randid]['e1']
        selected = train_df[train_df['e2'] == e2]
        if len(selected.index) >= 3:
            randids = random.sample(range(0, len(selected.index)),3)
            if len(randids) > len(set(randids)):
                continue
            else:
                e1 = selected.iloc[randids]['e1'].values
                r = selected.iloc[randids]['r'].values
                e1 = np.append(e1, e11)
                r = np.append(r, r11)
                e1_all.append(e1)
                r_all.append(r)
                e2_easy = []
                e2_all = []
                for round in range(4):
                    easy_next = easy_df[(easy_df['e1'] == e1[round]) & (easy_df['r'] == r[round])]['e2'].values
                    all_next = all_df[(all_df['e1'] == e1[round]) & (all_df['r'] == r[round])]['e2'].values
                    e2_easy.append(list(set(easy_next)))
                    e2_all.append(list(set(all_next)))
                answers_easy.append(list(set(e2_easy[0]).intersection(e2_easy[1], e2_easy[2], e2_easy[3])))
                answers_all.append(list(set(e2_all[0]).intersection(e2_all[1], e2_all[2], e2_all[3])))
                answers_hard.append([item for item in answers_all[-1] if item not in answers_easy[-1]])
                i = i + 1
    queries, easy = convert_format(e1_all, r_all, answers_easy, '4i')
    queries, hard = convert_format(e1_all, r_all, answers_hard, '4i')
    queries, all = convert_format(e1_all, r_all, answers_all, '4i')
    return queries, easy, hard, all

def create_5i(train_df, valid_df, test_df, nums, seed):
    random.seed(seed+3)
    easy_df = pd.concat([train_df, valid_df])
    all_df = pd.concat([train_df, valid_df, test_df])
    e1_all = []
    r_all = []
    answers_all = []
    answers_easy = []
    answers_hard = []
    i = 0
    while i < nums:
        randid = random.randrange(0, len(test_df.index))
        e2 = test_df.iloc[randid]['e2']
        r11 = test_df.iloc[randid]['r']
        e11 = test_df.iloc[randid]['e1']
        selected = train_df[train_df['e2'] == e2]
        if len(selected.index) >= 4:
            randids = random.sample(range(0, len(selected.index)),4)
            if len(randids) > len(set(randids)):
                continue
            else:
                e1 = selected.iloc[randids]['e1'].values
                r = selected.iloc[randids]['r'].values
                e1 = np.append(e1, e11)
                r = np.append(r, r11)
                e1_all.append(e1)
                r_all.append(r)
                e2_easy = []
                e2_all = []
                for round in range(5):
                    easy_next = easy_df[(easy_df['e1'] == e1[round]) & (easy_df['r'] == r[round])]['e2'].values
                    all_next = all_df[(all_df['e1'] == e1[round]) & (all_df['r'] == r[round])]['e2'].values
                    e2_easy.append(list(set(easy_next)))
                    e2_all.append(list(set(all_next)))
                answers_easy.append(list(set(e2_easy[0]).intersection(e2_easy[1], e2_easy[2], e2_easy[3], e2_easy[4])))
                answers_all.append(list(set(e2_all[0]).intersection(e2_all[1], e2_all[2], e2_all[3], e2_all[4])))
                answers_hard.append([item for item in answers_all[-1] if item not in answers_easy[-1]])
                i = i + 1
    queries, easy = convert_format(e1_all, r_all, answers_easy, '5i')
    queries, hard = convert_format(e1_all, r_all, answers_hard, '5i')
    queries, all = convert_format(e1_all, r_all, answers_all, '5i')
    return queries, easy, hard, all


def convert_format(e1, r, e2, gen_type):
    queries, answers = collections.defaultdict(set), collections.defaultdict(set)
    if gen_type == '4p':
        for i in range(len(e1)):
            query = (e1[i], (r[i][0], r[i][1], r[i][2], r[i][3]))
            queries[('e', ('r', 'r', 'r', 'r'))].add(query)
            answer = set(e2[i])
            answers[query] = set(answer)
        return queries, answers

    if gen_type == '5p':
        for i in range(len(e1)):
            query = (e1[i], (r[i][0], r[i][1], r[i][2], r[i][3], r[i][4]))
            queries[('e', ('r', 'r', 'r', 'r', 'r'))].add(query)
            answer = set(e2[i])
            answers[query] = set(answer)
        return queries, answers

    if gen_type == '4i':
        for i in range(len(e1)):
            query = ((e1[i][0], (r[i][0],)),(e1[i][1],(r[i][1], )),(e1[i][2],(r[i][2], )),(e1[i][3],(r[i][3], )))
            queries[(('e', ('r',)),('e', ('r', )), ('e',('r', )), ('e',('r', )))].add(query)
            answer = set(e2[i])
            answers[query] = set(answer)
        return queries, answers

    if gen_type == '5i':
        for i in range(len(e1)):
            query = ((e1[i][0], (r[i][0],)), (e1[i][1], (r[i][1],)), (e1[i][2], (r[i][2],)), (e1[i][3], (r[i][3],)), (e1[i][4], (r[i][4],)))
            queries[(('e', ('r', )), ('e', ('r', )), ('e', ('r', )), ('e', ('r', )), ('e', ('r', )))].add(query)
            answer = set(e2[i])
            answers[query] = set(answer)
        #pdb.set_trace()
        return queries, answers

def main(parser):
    create_nums = parser.test_num
    seed = parser.seed
    if create_nums > 0:
        #for dataset in ['FB15k-237-q2b', 'FB15k-q2b', 'NELL-q2b']:
        for dataset in ['NELL-q2b']:
            train_df, valid_df, test_df = load_data(parser.path, dataset)
            queries_4p, easy_4p,  hard_4p, all_4p = create_4p(train_df, valid_df, test_df, create_nums, seed)
            #pdb.set_trace()
            queries_5p, easy_5p,  hard_5p, all_5p = create_5p(train_df, valid_df, test_df, create_nums, seed)
            queries_4i, easy_4i,  hard_4i, all_4i = create_4i(train_df, valid_df, test_df, create_nums, seed)
            queries_5i, easy_5i,  hard_5i, all_5i = create_5i(train_df, valid_df, test_df, create_nums, seed)
        queries, easy, hard, all = {}, {}, {}, {}
        queries.update(queries_4p)
        queries.update(queries_5p)
        queries.update(queries_4i)
        queries.update(queries_5i)
        easy.update(easy_4p)
        easy.update(easy_5p)
        easy.update(easy_4i)
        easy.update(easy_5i)
        hard.update(hard_4p)
        hard.update(hard_5p)
        hard.update(hard_4i)
        hard.update(hard_5i)
        all.update(all_4p)
        all.update(all_5p)
        all.update(all_4i)
        all.update(all_5i)
        #pdb.set_trace()
        with open('data/NELL-q2b/test-queries-new.pkl', 'wb') as f:
            pickle.dump(queries, f)
        with open('data/NELL-q2b/test-easy-new.pkl', 'wb') as f:
            pickle.dump(easy, f)
        with open('data/NELL-q2b/test-hard-new.pkl', 'wb') as f:
            pickle.dump(hard, f)
        with open('data/NELL-q2b/test-all-new.pkl', 'wb') as f:
            pickle.dump(all, f)
if __name__ == '__main__':
    main(parse_args())