import collections
import csv

import torch
import numpy as np
import data_preprocessing.data_mapping as movie_mapping


def get_num_movie(fname):
    count = 0
    with open(fname.strip(), 'r') as fin:
        for _ in fin:
            count += 1
    return count


def get_user_ratings(fname, user_idx, k=None):
    user_ratings = []
    with open(fname, 'r') as file:
        rows = csv.reader(file, delimiter=':')
        for row in rows:
            u_id, m_id, rating = torch.LongTensor([int(row[0])]), int(row[1]), int(row[-2])
            if u_id == user_idx:
                user_ratings.append((m_id, rating))
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        if not k or len(user_ratings) < k:
            return user_ratings
        return user_ratings[:k]

def rate(model, user_idx, movie_idx, cat_idx):
    return model(user_idx, movie_idx, cat_idx)


def topkRate_have_watched(user_idx, k):
    topk_have_watched = get_user_ratings('data/train_data_sorted_vocab_id.dat', user_idx, k)
    m_new_vocab_id_to_m_name_and_id = movie_mapping.movie_new_vocab_id_to_movie_name_and_id()
    topk_have_watched = [(m_new_vocab_id_to_m_name_and_id[str(m_new_vocab_id)], rating) for (m_new_vocab_id, rating) in topk_have_watched]
    return topk_have_watched
    

def topK(model, user_idx, k=1):
    movie_id_to_cat_id = movie_mapping.generate_movie_id_to_cat_id_mapping()
    num_movie = get_num_movie('data/movie_id_mapping.dat')
    rating_per_movie = []
    for i in range(1, num_movie+1):
        movie_idx = torch.LongTensor([i])
        cat_idx = torch.LongTensor([[int(x) for x in list(movie_id_to_cat_id[str(i)])]])        
        rating = rate(model, user_idx, movie_idx, cat_idx)
        rating_per_movie.append((i, list(rating.cpu().numpy())[0]))
    rating_per_movie.sort(key = lambda x: x[1], reverse=True)
    return rating_per_movie[:k]


def topK_movie_id(model, user_idx, k=1):
    movie_id_to_cat_id = movie_mapping.generate_movie_id_to_cat_id_mapping()
    num_movie = get_num_movie('data/movie_id_mapping.dat')
    rating_per_movie = []
    m_new_vocab_id_to_m_id = movie_mapping.movie_new_vocab_id_to_movie_id()
    for i in range(1, num_movie+1):
        movie_idx = torch.LongTensor([i])
        cat_idx = torch.LongTensor([[int(x) for x in list(movie_id_to_cat_id[str(i)])]])        
        rating = rate(model, user_idx, movie_idx, cat_idx)
        rating_per_movie.append((m_new_vocab_id_to_m_id[str(i)], list(rating.cpu().numpy())[0]))
    rating_per_movie.sort(key = lambda x: x[1], reverse=True)
    return rating_per_movie[:k]


def topK_movie_name_and_id(model, user_idx, k=1):
    movie_id_to_cat_id = movie_mapping.generate_movie_id_to_cat_id_mapping()
    num_movie = get_num_movie('data/movie_id_mapping.dat')
    rating_per_movie = []
    m_new_vocab_id_to_m_name_and_id = movie_mapping.movie_new_vocab_id_to_movie_name_and_id()
    for i in range(1, num_movie+1):
        movie_idx = torch.LongTensor([i])
        cat_idx = torch.LongTensor([[int(x) for x in list(movie_id_to_cat_id[str(i)])]])        
        rating = rate(model, user_idx, movie_idx, cat_idx)
        rating_per_movie.append((m_new_vocab_id_to_m_name_and_id[str(i)], list(rating.cpu().numpy())[0]))
    rating_per_movie.sort(key = lambda x: x[1], reverse=True)
    return rating_per_movie[:k]    



def topK_optimized(model, user_idx, movie_id_to_cat_id, num_movie, k=1):
    rating_per_movie = []
    for i in range(1, num_movie+1):
        movie_idx = torch.LongTensor([i])
        cat_idx = torch.LongTensor([[int(x) for x in list(movie_id_to_cat_id[str(i)])]])        
        rating = rate(model, user_idx, movie_idx, cat_idx)
        rating_per_movie.append((i, list(rating.cpu().numpy())[0]))
    rating_per_movie.sort(key = lambda x: x[1], reverse=True)
    return rating_per_movie[:k]


def recallK_per_user(model, user_idx, k):
    target_user_rating = get_user_ratings('data/test_data_sorted_vocab_id.dat', user_idx, k)
    output_user_rating = topK(model, user_idx, k)
    if not target_user_rating or len(target_user_rating) < k:
        return None
    target_movies = set([movie for movie, rating in target_user_rating])
    num_overlap = 0
    for i in range(k):
        if output_user_rating[i][0] in target_movies:
            num_overlap += 1
    return num_overlap/k


def augmented_recallK_per_user(model, user_idx, k):
    target_user_rating = get_user_ratings('data/test_data_sorted_vocab_id.dat', user_idx, k)
    if not target_user_rating or len(target_user_rating) < k:
        return None
    targets = {movies[0]: i for i, movies in enumerate(target_user_rating)}
    movie_id_to_cat_id = movie_mapping.generate_movie_id_to_cat_id_mapping()
    rating_per_movie = []
    for i in targets.keys():
        movie_idx = torch.LongTensor([i])
        cat_idx = torch.LongTensor([[int(x) for x in list(movie_id_to_cat_id[str(i)])]]) 
        rating = rate(model, user_idx, movie_idx, cat_idx)
        rating_per_movie.append((i, list(rating.cpu().numpy())[0]))
    rating_per_movie.sort(key = lambda x: x[1], reverse=True)
    outputs = {movies[0]: i for i, movies in enumerate(rating_per_movie)}

    assert len(targets) == len(outputs)
    dist = 0
    max_dist = k * k if k % 2 == 0 else (k - 1) * (k + 1) / 2
    for movie_id in targets.keys():
        i, j = targets[movie_id], outputs[movie_id]
        dist += abs(i - j)
    return 1 - dist / max_dist

def augmented_recallK_batch_users(model, user_idx, k):
    recall_k = 0
    for i in user_idx:
        tmp = augmented_recallK_per_user(model, torch.LongTensor([i]), k)
        if tmp:
            recall_k += tmp
    return recall_k

