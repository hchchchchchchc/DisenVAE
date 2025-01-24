import math
import os
import sys
import pickle
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from preprocess import serial_asin_category
from extract_img_feature import get_img_feature_pickle
from support import RatingDataset
from tqdm import tqdm
import pandas as pd
import time
from support import serialize_user
from test import Validate
from myargs import get_args, args_tostring
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.set_device(2)


# CCFCRec
class DisenVAE(nn.Module):
    def __init__(self, args):
        super(DisenVAE, self).__init__()
        self.args = args
        self.attr_matrix = torch.nn.Parameter(torch.FloatTensor(args.attr_num, args.attr_present_dim))
        # 定义属性attribute注意力层
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        # 控制整个模型的激活函数
        self.h = nn.LeakyReLU()
        # 图像的映射矩阵
        self.image_projection = torch.nn.Parameter(torch.FloatTensor(4096, args.implicit_dim))
        self.sigmoid = torch.nn.Sigmoid()  # 将门控信号映射到[0, 1]之间
        self.moe_gate = nn.Sequential(nn.Linear(args.attr_present_dim * 3, args.attr_present_dim, bias=True),
                                      nn.Tanh(),
                                      nn.Linear(args.attr_present_dim, 2, bias=True),
                                      nn.Softmax(dim=-1))
        # self.fc1 = nn.Linear(args.attr_present_dim*2, args.cat_implicit_dim)
        self.fc1 = nn.Sequential(nn.Linear(args.attr_present_dim * 3, args.cat_implicit_dim),
                                 nn.BatchNorm1d(num_features=args.attr_present_dim),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(args.attr_present_dim * 2, args.cat_implicit_dim),
                                 nn.BatchNorm1d(num_features=args.attr_present_dim),
                                 nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(args.attr_present_dim * 3, args.cat_implicit_dim),
                                 nn.BatchNorm1d(num_features=args.cat_implicit_dim),
                                 nn.Tanh())
        self.mean_encoder_a_q = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.log_v_encoder_a_q = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.mean_encoder_c_q = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.log_v_encoder_c_q = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.mean_encoder_i = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.log_v_encoder_i = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.mean_encoder_a_p = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.log_v_encoder_a_p = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.mean_encoder_c_p = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.log_v_encoder_c_p = nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim),
            nn.BatchNorm1d(num_features=args.cat_implicit_dim),
            nn.Tanh()
        )
        self.decoder_a = nn.Sequential(nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim),
                                       nn.BatchNorm1d(num_features=args.cat_implicit_dim),
                                       nn.Tanh()
                                       )
        self.decoder_c = nn.Sequential(nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim),
                                       nn.BatchNorm1d(num_features=args.cat_implicit_dim),
                                       nn.Tanh()
                                       )
        self.decoder_p = nn.Sequential(nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim),
                                       nn.BatchNorm1d(num_features=args.cat_implicit_dim),
                                       nn.Tanh()
                                       )
        self.decoder_i = nn.Sequential(nn.Linear(in_features=args.implicit_dim, out_features=args.implicit_dim),
                                       nn.BatchNorm1d(num_features=args.cat_implicit_dim),
                                       nn.Tanh()
                                       )
        # user和item的嵌入层，可用预训练的进行初始化
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))
        # 定义生成层，将(q_v_a, u)的信息，共同生成 q_v_c， 生成包含协同信息的item嵌入
        self.gen_layer1 = nn.Linear(args.attr_present_dim*2, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)
        # 参数初始化
        self.__init_param__()

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        nn.init.xavier_normal_(self.image_projection)
        # 生成层初始化
        # user, item嵌入层的初始化, 没有预训练的情况下就初始化
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)

    def forward(self, attribute, image_feature, batch_size):
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1)+self.attr_b1.squeeze(), self.attr_W2)
        z_v_copy = z_v.repeat(batch_size, 1, 1)
        z_v_squeeze = z_v_copy.squeeze(dim=2).to(device)
        neg_inf = torch.full(z_v_squeeze.shape, -1e6).to(device)
        z_v_mask = torch.where(attribute != -1, z_v_squeeze, neg_inf)
        attr_attention_weight = torch.softmax(z_v_mask, dim=1)
        final_attr_emb = torch.matmul(attr_attention_weight, self.attr_matrix)
        p_v = torch.matmul(image_feature, self.image_projection)  # item的图像嵌入向量
        return final_attr_emb, p_v

def calc_kl_divergence(mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
    if mu1 is None or logvar1 is None:
        KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
    else:
        # JS divergence Gaussian version
        KLD = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
    if norm_value is not None:
        KLD = KLD / float(norm_value);
    return torch.mean(KLD)

def poe(mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    # precision of i-th Gaussian expert at point x
    T = 1. / var
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)
    return pd_mu, pd_logvar
def cal_loss_infonce(temperature, emb1, emb2, emb3):
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    emb3 = F.normalize(emb3, dim=1)
    pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / temperature)
    neg_score = torch.exp(torch.sum(emb1 * emb3, dim=1) / temperature)
    loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
    loss /= pos_score.shape[0]
    return loss
def train(model, train_loader, optimizer, valida, args, model_save_dir):
    print("model start train!")
    test_save_path = model_save_dir + "/result.csv"
    print("model train at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 写入超参数
    with open(model_save_dir + "/readme.txt", 'a+') as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write('\nsave dir:'+model_save_dir)
        f.write('\nmodel train time:'+(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    with open(test_save_path, 'a+') as f:
        f.write("loss,contrast_sum,hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n")
    save_index = 0
    for i_epoch in range(args.epoch):
        i_batch = 0
        batch_time = time.time()
        for user, item, item_genres, item_img_feature, neg_user, positive_item_list, negative_item_list, self_neg_list in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            # allocate memory cpu to gpu
            model = model.to(device)
            user = user.to(device)
            item = item.to(device)
            item_genres = item_genres.to(device)
            item_img_feature = item_img_feature.to(device)
            neg_user = neg_user.to(device)
            positive_item_list = positive_item_list.to(device)
            positive_item_emb = model.item_embedding[positive_item_list]
            negative_item_list = negative_item_list.to(device)
            negative_item_emb = model.item_embedding[negative_item_list]
            i_v = model.item_embedding[item]
            # run model
            a_v, c_v = model(item_genres, item_img_feature, user.shape[0])
            mean_i = model.mean_encoder_i(i_v)
            log_i = model.log_v_encoder_i(i_v)
            mean_a_q = model.mean_encoder_a_q(a_v)
            log_a_q = model.log_v_encoder_a_q(a_v)
            z_a = mean_a_q + torch.exp(log_a_q * 0.5) * torch.randn(mean_a_q.size()).to(device)

            h1 = model.fc2(torch.cat([a_v, c_v], dim=-1))
            mean_p = model.mean_encoder_a_p(h1)
            log_p = model.log_v_encoder_a_p(h1)

            mean_c_q = model.mean_encoder_c_q(c_v)
            log_c_q = model.log_v_encoder_c_q(c_v)
            z_c = mean_c_q + torch.exp(log_c_q * 0.5) * torch.randn(mean_c_q.size()).to(device)

            mean_p1, log_p1 = poe(torch.stack([mean_a_q, mean_c_q], dim=0),
                                  torch.stack([log_a_q, log_c_q], dim=0))
            z_p = mean_p1 + torch.exp(log_p1 * 0.5) * torch.randn(mean_p1.size()).to(device)


            user_emb = model.user_embedding[user]
            user_atten = model.moe_gate(torch.cat([user_emb, z_a, z_c], dim=-1))
            mean_q = torch.sum(user_atten.unsqueeze(-1) *
                               torch.cat([mean_a_q.unsqueeze(1), mean_c_q.unsqueeze(1)], dim=1),
                               dim=1, keepdim=False) * 0.5 + mean_p1 * 0.5
            log_q = torch.sum(user_atten.unsqueeze(-1) *
                              torch.cat([log_a_q.unsqueeze(1), log_c_q.unsqueeze(1)], dim=1),
                              dim=1, keepdim=False) * 0.5 + log_p1 * 0.5


            mean_mopoe = (mean_q + mean_i) / 2
            log_mopoe = (log_q + log_i) / 2
            z = mean_mopoe + torch.exp(log_mopoe * 0.5) * torch.randn(mean_mopoe.size()).to(device)

            decoder = model.decoder(model.fc3(torch.cat([z, a_v, c_v], dim=-1)))
            decoder_unsqueeze = decoder.unsqueeze(1)
            pos_contrast_mul = torch.sum(torch.mul(decoder_unsqueeze, positive_item_emb), dim=2) / (
                    args.tau * torch.norm(decoder_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10
            decoder_un2squeeze = decoder_unsqueeze.unsqueeze(dim=1)
            neg_contrast_mul = torch.sum(torch.mul(decoder_un2squeeze, negative_item_emb), dim=3) / (
                    args.tau * torch.norm(decoder_un2squeeze, dim=3) * torch.norm(negative_item_emb, dim=3))
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]
            contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))  # shape = [1024*10]
            contrast_sum = torch.sum(torch.sum(contrast_val, dim=1), dim=0) / contrast_val.shape[1]
            KLD = calc_kl_divergence(mean_mopoe, log_mopoe, mean_p, log_p) + (calc_kl_divergence(mean_i, log_i)
                                                                              + calc_kl_divergence(mean_c_q, log_c_q)
                                                                              + calc_kl_divergence(mean_a_q, log_a_q)) / 3
            decouple_contras_loss = cal_loss_infonce(args.tau, z_a, a_v, z_p) + cal_loss_infonce(args.tau, z_c, c_v,
                                                                                                 z_p)
            BCE = torch.square(i_v - decoder).sum(-1).mean()
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()
            y_uv_i = torch.mul(i_v, user_emb).sum(dim=1)
            y_kv_i = torch.mul(i_v, neg_user_emb).sum(dim=1)
            y_ukv_i = -logsigmoid(y_uv_i - y_kv_i).sum()
            y_uv2 = torch.mul(decoder, user_emb).sum(dim=1)
            y_kv2 = torch.mul(decoder, neg_user_emb).sum(dim=1)
            y_ukv2 = -logsigmoid(y_uv2 - y_kv2).sum()
            total_loss = 10 * (KLD + BCE) + y_ukv2 + y_ukv_i + contrast_sum + decouple_contras_loss

            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)
            total_loss.backward()
            optimizer.step()
            i_batch += 1
            if i_batch % args.save_batch_time == 0:
                model.eval()
                print("[{},/13931603]total_loss:,{},{},s".format(i_batch*1024, total_loss.item(), int(time.time()-batch_time)))
                with torch.no_grad():
                    hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20, x, it_idx = valida.start_validate(model)
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)  # perplexity 需要小于样本数
                x_embedded = tsne.fit_transform(x)
                colors = ['red'] * it_idx + ['green'] * it_idx + ['blue'] * it_idx  # 前 n 个红色，中间 n 个绿色，后 n 个蓝色

                # 可视化结果
                plt.figure(figsize=(10, 8))
                plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=colors, s=50, alpha=0.6)
                plt.title('t-SNE Visualization with Grouped Colors')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.savefig(model_save_dir + '/epoch_' + str(i_epoch) + "batch_" + str(i_batch) + "t-sne.png")
                with open(test_save_path, 'a+') as f:
                    f.write("{},{},{},{},{},{},{},{}\n".format(total_loss.item(), contrast_sum, hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20))
                # 保存模型
                batch_time = time.time()
                save_index += 1
                torch.save(model.state_dict(), model_save_dir + '/' + str(save_index)+".pt")


if __name__ == '__main__':
    # result save dir
    save_dir = 'result/' + time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    os.makedirs(save_dir)
    # args
    args = get_args()
    print("progress start at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/validate_rating.csv'
    train_df = pd.read_csv(train_path)
    total_user_set = train_df['reviewerID']
    user_ser_dict = serialize_user(total_user_set)
    asin_category_int_map, category_ser_map = serial_asin_category()
    img_feature_dict = get_img_feature_pickle()
    # write internal variable
    with open(save_dir+"/save_dict.pkl", "wb") as file:
        save_dict = {'img_feature_dict': img_feature_dict, 'asin_category_int_map': asin_category_int_map,
                     'category_ser_map_len': category_ser_map.__len__(), 'user_ser_dict': user_ser_dict}
        pickle.dump(save_dict, file)
    # load dataset
    dataSet = RatingDataset(train_df, img_feature_dict, asin_category_int_map, category_ser_map.__len__(),
                            user_ser_dict, args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.all_item_number
    item_ser_dict = dataSet.item_serialize_dict
    train_loader = torch.utils.data.DataLoader(dataSet, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("模型超参数:", args_tostring(args))
    myModel = DisenVAE(args)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.learning_rate, weight_decay=0.1)
    validator = Validate(validate_csv=vliad_path, user_serialize_dict=user_ser_dict, item_serialize_dict=item_ser_dict, img=img_feature_dict,
                         genres=asin_category_int_map, category_num=category_ser_map.__len__())
    train(myModel, train_loader, optimizer, validator, args, save_dir)

