import torch
import csv
import os
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoTower(nn.Module):
    def __init__(self, 
      num_user, 
      num_movie, 
      num_category, 
      user_layer_size = [32], 
      movie_layer_size = [64, 32], 
      uid_embedding_size = 64,
      mid_embedding_size = 64,
      cat_embedding_size = 16, 
      popularity_hashing = False, 
      uid_ph = [0.7, 0.9], 
      mid_ph = [0.7, 0.9]
    ):
        super().__init__() 
        self.num_user = num_user
        self.num_movie = num_movie
        self.num_category = num_category
        self.uid_embedding_size = uid_embedding_size
        self.mid_embedding_size = mid_embedding_size
        self.cat_embedding_size = cat_embedding_size
        self.movie_layer_size = movie_layer_size
        self.popularity_hashing = popularity_hashing
        self.uid_ph = uid_ph
        self.mid_ph = mid_ph
        self.num_uid_hashing = None
        self.num_uid_popular = None
        self.num_mid_hashing = None
        self.num_mid_popular = None

        # user embedding table
        if self.popularity_hashing:
            self.num_uid_hashing = int((self.num_user+1) * self.uid_ph[0])
            self.num_uid_popular = int(self.num_uid_hashing * self.uid_ph[1])
            self.user_embedding = nn.Embedding(self.num_uid_hashing+1, uid_embedding_size)
        else:
            self.user_embedding = nn.Embedding(self.num_user+1, uid_embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        # movie embedding table
        if self.popularity_hashing:
            self.num_mid_hashing = int((num_movie+1) * self.mid_ph[0])
            self.num_mid_popular = int(self.num_mid_hashing * self.mid_ph[1])
            self.movie_embedding = nn.Embedding(self.num_mid_hashing+1, mid_embedding_size)
        else:
            self.movie_embedding = nn.Embedding(num_movie+1, mid_embedding_size)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        # category embedding table
        self.category_embedding = nn.Embedding(num_category+2, cat_embedding_size)
        nn.init.xavier_uniform_(self.category_embedding.weight)

        #user embedding MLP part
        self.linear_user = nn.Linear(uid_embedding_size, user_layer_size[0])
        
        #movie embedding MLP part
        all_dims = [self.mid_embedding_size + self.cat_embedding_size] + movie_layer_size
        for i in range(1, len(movie_layer_size)+1):
            setattr(self, 'linear_movie_' + str(i), nn.Linear(all_dims[i-1], all_dims[i]))
    

    def forward(self, user_idx, movie_idx, category_idx_lst):
        # rehash the user_idx and movie_idx if popularity hashing is set
        if self.popularity_hashing:
            batches = len(user_idx)
            user_idx_ph = []
            movie_idx_ph = []
            for i in range(batches):
                if user_idx[i] < self.num_uid_popular:
                    user_idx_ph.append(user_idx[i])
                else:
                    ph_user_idx = user_idx[i] % (self.num_uid_hashing-self.num_uid_popular) + self.num_uid_popular
                    user_idx_ph.append(ph_user_idx)
                if movie_idx[i] < self.num_mid_popular:
                    movie_idx_ph.append(movie_idx[i])
                else:
                    ph_movie_idx = movie_idx[i] % (self.num_mid_hashing-self.num_mid_popular) + self.num_mid_popular
                    movie_idx_ph.append(ph_movie_idx)
            user_idx = torch.LongTensor([int(x) for x in user_idx_ph]).to(DEVICE)
            movie_idx = torch.LongTensor([int(x) for x in movie_idx_ph]).to(DEVICE)
            
        user_feat_emb = self.user_embedding(user_idx)
        movie_feat_emb = self.movie_embedding(movie_idx)
        summed_category_emb = []
        for batch in range(len(category_idx_lst)):
            category_idx_lst_single_batch = category_idx_lst[batch]
            n_cat = 0
            for idx in category_idx_lst_single_batch:
                if idx == -1:
                    break
                n_cat += 1
            category_idx_lst_single_batch_filtered = category_idx_lst_single_batch[0:n_cat]
            category_feat_emb_single_batch = self.category_embedding(category_idx_lst_single_batch_filtered)
            # sum_pooling for all categories embedding for a movie
            summed_category_emb_single_batch = torch.sum(category_feat_emb_single_batch, 0)
            summed_category_emb.append(summed_category_emb_single_batch)
        summed_category_emb = torch.cat(summed_category_emb, dim=0)
        summed_category_emb = torch.reshape(summed_category_emb, (len(category_idx_lst), self.cat_embedding_size))
        concat_movie_category_emb = torch.cat((movie_feat_emb, summed_category_emb), dim = 1)        

        #user MLP
        user_y_deep = user_feat_emb
        user_y_deep = self.linear_user(user_y_deep)
        user_y_deep = F.relu(user_y_deep)

        #movie MLP
        movie_y_deep = concat_movie_category_emb
        
        for i in range(1, len(self.movie_layer_size)+1):
            movie_y_deep = getattr(self, 'linear_movie_' + str(i))(movie_y_deep)
            movie_y_deep = F.relu(movie_y_deep)
        
        # calculate dot product
        mul_user_movie = torch.mul(user_y_deep, movie_y_deep)
        sum_user_movie = torch.sum(mul_user_movie, dim = 1)
        
        user_square = torch.mul(user_y_deep, user_y_deep)
        user_sum = torch.sum(user_square, 1)
        user_square_root = torch.sqrt(user_sum)
        
        movie_square = torch.mul(movie_y_deep, movie_y_deep)
        movie_sum = torch.sum(movie_square, 1)
        movie_square_root = torch.sqrt(movie_sum)
        value_mul= torch.mul(user_square_root, movie_square_root)

        output = torch.div(sum_user_movie, value_mul)

        return output