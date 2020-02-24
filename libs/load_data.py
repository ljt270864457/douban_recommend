#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/24 8:25 PM
# @Author  : liujiatian
# @File    : load_data.py

import os

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORI_DATA_DIR = os.path.join(ROOT_DIR, 'dataset/orign_data')
MODEL_DATA_DIR = os.path.join(ROOT_DIR, 'dataset/model_data')
CLEANED_DATA_DIR = os.path.join(ROOT_DIR, 'dataset/cleaned_data')


def load_rating_data():
    '''
    获取打分数据,将6分以上的作为用户喜欢的
    :return:
    '''
    rating_path = os.path.join(CLEANED_DATA_DIR, 'rating.csv')
    df = pd.read_csv(rating_path, header=None)
    df.columns = ['user_id', 'movie_id', 'comment_time', 'rating']
    df_like = df[df['rating'] > 6]
    print(f'用户数量:{len(df_like["user_id"].drop_duplicates())}')
    print(f'电影数量:{len(df_like["movie_id"].drop_duplicates())}')
    rating_data = df_like.to_dict(orient='records')
    return rating_data


def load_movie():
    result = {}
    movie_path = os.path.join(CLEANED_DATA_DIR, 'movie.csv')
    df = pd.read_csv(movie_path, usecols=['movie_id', 'movie_name'])
    records = df.to_dict(orient='records')
    for each in records:
        movie_id = each.get('movie_id')
        movie_name = each.get('movie_name')
        result[movie_id] = movie_name
    return result