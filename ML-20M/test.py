import pandas as pd
import torch
import numpy as np
import random
import time
import os

from torchmetrics.functional.segmentation import mean_iou

from myargs import get_args
from support import serialize_all_item, genres_item_dic
import matplotlib.pyplot as plt

from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def poe(mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    # precision of i-th Gaussian expert at point x
    T = 1. / var
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)
    return pd_mu, pd_logvar

def poe_fusion(mus, logvars, weights=None):
    num_samples = mus[0].shape[0]
    mus = torch.cat((mus, torch.zeros(1, num_samples,
                    128).to(device)),
                    dim=0)
    logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                    128).to(device)),
                    dim=0)
    #mus = torch.cat(mus, dim=0);
    #logvars = torch.cat(logvars, dim=0);
    mu_poe, logvar_poe = poe(mus, logvars)
    return mu_poe, logvar_poe

def get_similar_user_speed(model, genres, img, k, item_id):
    user_embedding = model.user_embedding
    item_embedding = model.item_embedding
    i_v = item_embedding[item_id].unsqueeze(0)
    user_idx = torch.tensor(list(range(user_embedding.shape[0])))
    user_idx = user_idx.to(device)
    # [138493*64]
    user_emb = user_embedding[user_idx]
    genres = genres.unsqueeze(dim=0)
    img = img.unsqueeze(dim=0)
    # 输入到模型中
    attr_present = model.attr_matrix(genres)
    attr_tmp1 = model.h(torch.matmul(attr_present, model.attr_W1.T) + model.attr_b1)
    attr_attention_b = model.softmax(torch.matmul(attr_tmp1, model.attr_W2))
    a_v = torch.matmul(attr_attention_b.transpose(1, 2), attr_present).squeeze(0)  # z_v是属性经过注意力加权融合后的向量
    c_v = torch.matmul(img, model.image_projection)  # item的图像嵌入向量
    mean_a = model.mean_encoder_a_q(a_v)
    log_a = model.log_v_encoder_a_q(a_v)

    mean_c = model.mean_encoder_c_q(c_v)
    log_c = model.log_v_encoder_c_q(c_v)

    mean_i = model.mean_encoder_i(i_v)

    mean_p, log_p = poe(torch.stack([mean_a, mean_c], dim=0),
                               torch.stack([log_a, log_c], dim=0))

    user_atten = model.moe_gate(torch.cat([user_emb, mean_a.expand_as(user_emb), mean_c.expand_as(user_emb)], dim=-1))
    mean_f = torch.sum(user_atten.unsqueeze(-1) *
                        torch.cat([mean_a.unsqueeze(1), mean_c.unsqueeze(1)], dim=1),
                        dim=1, keepdim=False) * 0.5 + mean_p * 0.5

    mean = (mean_f + mean_i.expand_as(mean_f)) / 2
    z = mean

    h3 = model.fc3(torch.cat([z, a_v.expand_as(z), c_v.expand_as(z)], dim=-1))
    rec = model.decoder(h3)

    ratings = torch.mul(user_emb, rec).sum(dim=-1)
    index = torch.argsort(-ratings)
    return index[0:k].cpu().detach().numpy().tolist(), mean_a, mean_c, mean_p

def hr_at_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    inter = set(groundtruth_user).intersection(set(recommend_users))
    return len(inter)


def dcg_k(r):
    r = np.asarray(r)
    val = np.sum((np.power(2, r) - 1) / (np.log2(np.arange(1+1, r.size + 2))))
    return val


def ndcg_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    ratings = []
    ndcg = 0.0
    for u in recommend_users:
        if u in groundtruth_user:
            ratings.append(1.0)
        else:
            ratings.append(0.0)
    ratings_ideal = sorted(ratings, reverse=True)
    ideal_dcg = dcg_k(ratings_ideal)
    if ideal_dcg != 0:
        ndcg = (dcg_k(ratings) / ideal_dcg)
    return ndcg


class Validate:
    def __init__(self, validate_csv, user_serialize_dict, img, genres):
        print("validate class init")
        validate_csv = pd.read_csv(validate_csv, dtype={'userId': int, 'movieId': int, 'rating': float})
        self.item = set(validate_csv['movieId'])
        self.item_user_dict = {}
        # 构建完成 item->user dict
        for it in self.item:
            users = validate_csv[validate_csv['movieId'] == it]['userId']
            users = [user_serialize_dict.get(u) for u in users]
            self.item_user_dict[it] = users
        self.img_dict = img
        self.genres_dict = genres
        self.item_serialize_dict = serialize_all_item(genres.keys())
        self.genres_item_dict = genres_item_dic(self.item_serialize_dict)
        self.all_item_number = len(self.item_serialize_dict.keys())


    def start_validate(self, model):
        # 开始评估
        hr_hit_cnt_5, hr_hit_cnt_10, hr_hit_cnt_20 = 0, 0, 0
        ndcg_sum_5, ndcg_sum_10, ndcg_sum_20 = 0.0, 0.0, 0.0
        max_k = 20
        it_idx = 0
        attr = torch.Tensor()
        image = torch.Tensor()
        common = torch.Tensor()
        for it in self.item:
            # 输出
            model = model.to(device)  # move to cpu
            item_id = torch.tensor(self.item_serialize_dict[it])
            item_id = item_id.to(device)
            genres = torch.tensor(self.genres_dict.get(it))
            img_feature = torch.tensor(self.img_dict.get(it))
            genres = genres.to(device)
            img_feature = img_feature.to(device)

            with torch.no_grad():
                recommend_users, mean_a, mean_c, mean_p = get_similar_user_speed(model, genres, img_feature, max_k, item_id)
            attr = torch.concat([attr, mean_a.cpu()], dim=0)
            image = torch.concat([image, mean_c.cpu()], dim=0)
            common = torch.concat([common, mean_p.cpu()], dim=0)



            # 计算hr指标
            # 计算p@k指标
            hr_hit_cnt_5 += hr_at_k(it, recommend_users, self.item_user_dict, 5)
            hr_hit_cnt_10 += hr_at_k(it, recommend_users, self.item_user_dict, 10)
            hr_hit_cnt_20 += hr_at_k(it, recommend_users, self.item_user_dict, 20)
            # 计算NDCG指标
            ndcg_sum_5 += ndcg_k(it, recommend_users, self.item_user_dict, 5)
            ndcg_sum_10 += ndcg_k(it, recommend_users, self.item_user_dict, 10)
            ndcg_sum_20 += ndcg_k(it, recommend_users, self.item_user_dict, 20)
            # print("评估进度:", it_idx, "/", len(item))
            it_idx += 1
        x = torch.concat([attr, image, common], dim=0).squeeze(1).cpu()


        # 标注每个点
        item_len = len(self.item)
        hr_5 = hr_hit_cnt_5 / (item_len * 5)
        hr_10 = hr_hit_cnt_10 / (item_len * 10)
        hr_20 = hr_hit_cnt_20 / (item_len * 20)
        ndcg_5 = ndcg_sum_5/item_len
        ndcg_10 = ndcg_sum_10/item_len
        ndcg_20 = ndcg_sum_20/item_len
        print("hr@5:", "hr_10:", "hr_20:", 'ndcg@5', 'ndcg@10', 'ndcg@20')
        print(hr_5, ',', hr_10, ',', hr_20, ',', ndcg_5, ',', ndcg_10, ',', ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20, x, it_idx


if __name__ == '__main__':
    # 参数解析器
    from support import read_img_feature, read_genres, serialize_user
    from model import DisenVAE

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    train_df = pd.read_csv("data/train_rating.csv", dtype={'userId': int, 'movieId': int, 'neg_user_id': int})
    total_user_set = train_df['userId']
    user_serialize_dict = serialize_user(total_user_set)
    img_feature = read_img_feature('data/img_feature.csv')
    movie_onehot = read_genres('data/movies_onehot.csv')
    myModel = DisenVAE(args)
    validator = Validate(validate_csv='data/validate_rating.csv', user_serialize_dict=user_serialize_dict,
                         img=img_feature, genres=movie_onehot)
    print('---------数据集加载完毕，开始测试----------------')
    test_result_name = 'test_result.csv'
    with open(test_result_name, 'a+') as f:
        f.write("hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n")
    load_dir = '/root/data1/hc/CCFCRec/ML-20M/result/poe+moe+cvae/2025-01-16_10:53:45/'
    load_array = ['1batch_12000']
    for model in load_array:
        myModel.load_state_dict(torch.load(load_dir+'/epoch_'+model+'.pt', map_location=device))
        hr5, hr_10, hr_20, n_5, n_10, n_20 = validator.start_validate(myModel)
        with open(test_result_name, 'a+') as f:
            f.write("{},{},{},{},{},{}\n".format(hr5, hr_10, hr_20, n_5, n_10, n_20))