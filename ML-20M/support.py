import gc
import pickle
import random
import os
import sys
import time
import numpy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import ast
from tqdm import tqdm


# 返回一个dict， userId: [positive_sample(list), negative_sample(list(list))]
def read_user_positive_negative_movies(user_positive_movie_csv, refresh=False):
    pkl_name = 'pkl/user_pn_dict.pkl'
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    if (os.path.exists(pkl_name) is True) and (refresh is False):
        pkl_file = open(pkl_name, 'rb')
        data = pickle.load(pkl_file)
        return data['user_pn_dict']
    user_position_dict = {}
    last_user = -1
    user_position_dict[last_user] = [-1, -1]
    for index, row in tqdm(user_positive_movie_csv.iterrows()):
        u = row['userId']
        if u != last_user:
            user_position_dict[u] = [index, index]
            user_position_dict[last_user] = [user_position_dict.get(last_user)[0], index-1]
            last_user = u
    # 更新最后一项
    user_position_dict[last_user] = [user_position_dict.get(last_user)[0], user_positive_movie_csv.__len__()-1]
    with open(pkl_name, 'wb') as file:
        pickle.dump({'user_pn_dict': user_position_dict}, file)
    return user_position_dict

def user_pos_neg_dict(user_position_dict, user_pos_neg_movie_df, positive_number, neg_num, user_serialize_dict, item_serialize_dict):

    user_positive_dict = {}
    user_negtive_dict = {}
    for k in user_position_dict:
        if k >0:
           position_arr = user_position_dict.get(k)
           positive_segment = user_pos_neg_movie_df.loc[position_arr[0]: position_arr[1]]
           positive_movie_df = pd.DataFrame.sample(positive_segment, n=positive_number, replace=True)
           positive_movie_list = list(positive_movie_df['positive_movies'])
           positive_movie_list = [item_serialize_dict.get(item) for item in positive_movie_list]
           user_positive_dict[user_serialize_dict.get(k)] = positive_movie_list
           neg_list = []
           negative_movie_list = positive_movie_df['negative_movies']
           for neg in negative_movie_list:
               tmp_neg_list = list(map(int, neg[1:-1].split(",")))
               tmp_neg_ser_list = [item_serialize_dict.get(item) for item in tmp_neg_list]
               # 插入一条抽样
               neg_list.append(list(np.random.choice(tmp_neg_ser_list, neg_num, replace=True)))
           user_negtive_dict[user_serialize_dict.get(k)] = neg_list
    return user_positive_dict, user_negtive_dict

def read_img_feature(img_feature_csv):
    df = pd.read_csv(img_feature_csv, dtype={'feature': object, 'movie_id': int})
    img_feature_dict = {}
    for index, row in df.iterrows():
        item = row['movie_id']
        feature = list(map(float, row['feature'][1:-1].split(",")))
        img_feature_dict[item] = feature
    return img_feature_dict


def read_genres(genres_csv):
    df = pd.read_csv(genres_csv, dtype={'movieId': int})
    genres_dict = {}
    for index, row in df.iterrows():
        item = row['movieId']
        genres = list(map(int, row['genres_onehot'][1:-1].split(',')))
        genres_dict[item] = genres
    return genres_dict

def genres_item_dic(item_remap):
    data = pd.read_csv('data/movies_onehot.csv', dtype={'movieId': int})

    # 创建一个字典来存储类别和对应的电影 ID
    categories = {}
    item_serialize_dict = item_remap

    # 遍历每一行，提取电影 ID 和类别
    for index, row in data.iterrows():
        movie_id = row['movieId']
        genres = ast.literal_eval(row['genres_onehot'])  # 将字符串转换为列表
        for genre in genres:
            if genre != 18:  # 忽略缺省值
                if genre not in categories:
                    categories[genre] = []
                categories[genre].append(item_serialize_dict.get(movie_id))

    # 输出每个类别及其对应的电影 ID
    with open('data/categories.pkl', 'wb') as f:
        pickle.dump(categories, f)
    return categories

def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    # key: user原始下标，value: user有序下标
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict


# 输入user和item的set，输出user和item从1到n有序的字典
def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict

def serialize_all_item(item_set):
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict

def attr_sample(genres, genres_item_dict, sample_number, item_number):
    attr_sample = []
    for i in range(len(genres)):
        if genres[i] != 18:
            attr_sample.append(random.sample(genres_item_dict.get(genres[i]), sample_number))
        else:
            attr_sample.append([item_number for i in range(sample_number)])
    return attr_sample

class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, user_positive_movie_csv, img_features, genres, user_serialize_dict, item_pn_csv,
                 positive_number, negative_number, sample_number):
        self.train_csv = train_csv
        # 读hypergraph数据
        # 读其他内容
        self.img_feature_dict = img_features
        self.genres_dict = genres
        self.user_pos_neg_movie_df = pd.read_csv(user_positive_movie_csv, dtype={'userId': np.int32, 'positive_movies': np.int32})
        self.item_pn_df = pd.read_csv(item_pn_csv)
        # print(self.item_pn_df)
        self.user_position_dict = read_user_positive_negative_movies(self.user_pos_neg_movie_df)

        self.user = self.train_csv["userId"]
        self.neg_user = self.train_csv['neg_user_id']
        self.item = self.train_csv["movieId"]
        self.rating = self.train_csv["rating"]
        # 序列化user和item
        self.user_serialize_dict = user_serialize_dict
        self.item_serialize_dict = serialize_all_item(genres.keys())
        self.genres_serialize_dict = {self.item_serialize_dict[k]: v for k, v in self.genres_dict.items()}
        self.img_serialize_dict = {self.item_serialize_dict[k]: v for k, v in self.img_feature_dict.items()}
        # print(len(self.item_serialize_dict))
        self.genres_item_dict = genres_item_dic(self.item_serialize_dict)
        self.all_item_number = len(self.item_serialize_dict.keys())
        # 返回个数时，返回全集的user数和训练集的item数
        self.user_number = len(user_serialize_dict)
        self.item_number = len(set(self.item))
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.sample_number = sample_number

        self.user_pos_dict, self.user_neg_dict = user_pos_neg_dict(self.user_position_dict, self.user_pos_neg_movie_df,
                                                                   positive_number, negative_number, self.user_serialize_dict, self.item_serialize_dict)
        print("整个数据集的user个数为:", self.user_number, "train_set中的用户数目为:", len(set(self.user)))

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]
        neg_user = self.neg_user[index]
        # 处理 item genres
        genres = self.genres_dict.get(item)
        # 基于 attr 采样
        # attr_sample = []
        # for i in range(len(genres)):
        #     if genres[i] != 18:
        #         attr_sample.append(random.sample(self.genres_item_dict.get(genres[i]), self.sample_number))
        #     else:
        #         attr_sample.append([self.all_item_number for i in range(self.sample_number)])

        # 处理 item feature
        img_feature = self.img_feature_dict.get(item)
        # 处理 positive items
        # 直接存储df的位置，user -> 从哪里到哪里

        self_negative_list = self.item_pn_df['negative_movies'][index]
        # self neg list 完成 序列化
        tmp_neg_list = list(map(int, self_negative_list[1:-1].split(",")))
        tmp_neg_ser_list = [self.item_serialize_dict.get(item) for item in tmp_neg_list]
        # 插入一条抽样
        # self_neg_list = list(np.random.choice(tmp_neg_ser_list, self.negative_number, replace=True))
        # coll neg完成序列化
        # neg_list = []
        # for neg in negative_movie_list:
        #     tmp_neg_list = list(map(int, neg[1:-1].split(",")))
        #     tmp_neg_ser_list = [self.item_serialize_dict.get(item) for item in tmp_neg_list]
        #     # 插入一条抽样
        #     neg_list.append(list(np.random.choice(tmp_neg_ser_list, self.negative_number, replace=True)))
        # 对当前item进行抽样
        # user，item id进行序列化
        user = self.user_serialize_dict.get(user)
        neg_user = self.user_serialize_dict.get(neg_user)
        item = self.item_serialize_dict.get(item)
        positive_movie_list = self.user_pos_dict.get(user)
        # print(type(positive_movie_list))
        # print(len(positive_movie_list))
        # positive_genre_list = [self.genres_serialize_dict.get(i) for i in positive_movie_list]
        # positive_img = [self.img_serialize_dict.get(i) for i in positive_movie_list]
        negative_movie_list = self.user_neg_dict.get(user)
        # negative_genre_list = [[self.genres_serialize_dict.get(i) for i in negative_movie_list[j]] for j in range(len(negative_movie_list))]
        # negative_img = [[self.genres_serialize_dict.get(i) for i in negative_movie_list[j]] for j in range(len(negative_movie_list))]
        # print(len(negative_img))
        # for j in range(len(negative_movie_list)):
        #     negative_genre_list.append([self.genres_serialize_dict.get(i) for i in negative_movie_list[j]])
        #     negative_img.append(self.genres_serialize_dict.get(i) for i in negative_movie_list[j])
        # 序列化positive_movie_list
        # positive_movie_list = [self.item_serialize_dict.get(item) for item in positive_movie_list]
        return torch.tensor(user), torch.tensor(item), torch.tensor(genres), torch.tensor(img_feature), \
               torch.tensor(neg_user), torch.tensor(positive_movie_list), torch.tensor(negative_movie_list), \
               torch.tensor(tmp_neg_ser_list)