#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/20 5:24 PM
# @Author  : liujiatian
# @File    : user_cf.py

import math

from itertools import combinations
from collections import Counter

import numpy as np
from libs.load_data import load_rating_data, load_movie

import sys

sys.path.append('../../')

'''
item-cf算法描述

1.计算各个电影之间的相似度
2.找到与用户喜欢的电影最相似的n个电影


1.构建
电影-用户正排表
 -> movie1:[user1,user2,user3]
 -> movie2:[user1,user4,user8]
 
用户-电影倒排表
-> user1:[movie1,movie3,movie5]
-> user2:[movie1,movie6,movie10]

2.根据用户倒排表计算共现矩阵
[
((movie1,movie3),1),
((movie2,movie4),3),
]

3.计算相似度
- jaccard
A∩B/(A∪B)
- 余弦相似度
A∩B/sqrt(|A|*|B|)
- 皮尔逊相关系数

4.给用户推荐最喜欢的n个商品
    - 找到用户喜欢的商品列表中，与每个物品最相似的K个商品
    - rui = ∑w*r
'''
# TODO 评价指标的构建

def construct_ascending_data(rating_data):
    '''
    构建正排表和倒排表
    :param rating_data 打分数据
    :return:
    '''
    # 正排表
    movie_user = {}
    # 倒排表
    user_movie = {}
    for each in rating_data:
        user_id = each.get('user_id')
        movie_id = each.get('movie_id')
        # 构建正排表
        if movie_id not in movie_user:
            movie_user[movie_id] = [user_id]
        else:
            movie_user[movie_id].append(user_id)
        # 构建倒排表
        if user_id not in user_movie:
            user_movie[user_id] = [movie_id]
        else:
            user_movie[user_id].append(movie_id)
    return movie_user, user_movie


def construct_co_matrix(user_movie):
    '''
    根据正排表构建共现矩阵
    :param user_movie:
    :return:
    '''
    co_related_list = []
    for each in user_movie.values():
        if len(each) < 2:
            continue
        combins = combinations(sorted(each), 2)
        co_related_list.extend(combins)
    return co_related_list


# 计算相似度
def calc_jaccard(movie_user, item_id1, item_id2):
    '''
    jacard相似度
    :param movie_user:
    :param item_id1:
    :param item_id2:
    :return:
    '''
    user_arr1 = np.array(movie_user.get(item_id1, []))
    user_arr2 = np.array(movie_user.get(item_id2, []))
    # 交集元素
    intersect = np.intersect1d(user_arr1, user_arr2)
    # 并集元素
    union = np.union1d(user_arr1, user_arr2)
    try:
        sim_jaccard = len(intersect) / len(union)
    except ZeroDivisionError:
        sim_jaccard = 0
    return sim_jaccard


def calc_cosine(movie_user, item_id1, item_id2):
    '''
    计算余弦相似度
    :param movie_user:
    :param item_id1:
    :param item_id2:
    :return:
    '''
    user_arr1 = np.array(movie_user.get(item_id1, []))
    user_arr2 = np.array(movie_user.get(item_id2, []))
    # 交集元素
    intersect = np.intersect1d(user_arr1, user_arr2)
    try:
        sim_cosine = len(intersect) / math.sqrt(len(user_arr1) * len(user_arr2))
    except ZeroDivisionError:
        sim_cosine = 0
    return sim_cosine


def construct_sim_matrix(movie_user, co_related_list, method='cosine'):
    '''
        sim_jaccard_co_movie格式
        {
        movie1:{
        movie2:sim1,
        movie3:sim2
        }
    '''
    counter = Counter(co_related_list)
    sim_co_movie = {}
    counter_dict = dict(counter)
    for co_movie, count in counter_dict.items():
        movie1, movie2 = co_movie
        if method == 'cosine':
            sim = calc_cosine(movie_user, movie1, movie2)
        else:
            sim = calc_jaccard(movie_user, movie1, movie2)
        if movie1 not in sim_co_movie:
            sim_co_movie[movie1] = {movie2: sim}
        else:
            sim_co_movie[movie1][movie2] = sim

        if movie2 not in sim_co_movie:
            sim_co_movie[movie2] = {movie1: sim}
        else:
            sim_co_movie[movie2][movie1] = sim
    return sim_co_movie


def get_most_sim(sim_co_movie, movie_id, K=40):
    '''
    根据电影ID，获取与之最相近的K个电影
    '''
    movie_sim_all = sim_co_movie.get(movie_id) or {}
    result = sorted(movie_sim_all.items(), key=lambda x: x[1], reverse=True)[:K]
    return result


def recommend(user_id, user_movie, sim_co_movie, K=40):
    rank = {}
    history_movie = user_movie.get(user_id)
    for movie_id in history_movie:
        tmp_list = get_most_sim(sim_co_movie, movie_id, K)
        for _id, sim in tmp_list:
            if _id in history_movie:
                continue
            if _id not in rank:
                rank[_id] = sim
            else:
                rank[_id] += sim
    return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:K]


def main():
    rating_data = load_rating_data()
    print(len(rating_data))
    movie_data = load_movie()
    movie_user, user_movie = construct_ascending_data(rating_data)
    co_related_list = construct_co_matrix(user_movie)
    sim_co_movie = construct_sim_matrix(movie_user, co_related_list)
    result = recommend(4, user_movie, sim_co_movie)
    print('=' * 10)
    for i in result:
        print(f'用户喜欢【【{movie_data.get(i[0])}】,权重为:{i[1]}')
    return result


if __name__ == '__main__':
    main()
