import csv
import collections


MOVIE_INFO_FILE = 'data/movie_id_mapping.dat'
MOVIE_TRAIN_FILE = 'data/train_data_sorted_vocab_id.dat'

def movie_new_vocab_id_to_movie_name():
    movie_new_vocab_id_to_movie_name_mapping = {}
    with open(MOVIE_INFO_FILE, 'r') as file:
        rows = csv.reader(file, delimiter=':')
        for row in rows:
            new_vocab_id, movie_name = row[1], row[3]
            movie_new_vocab_id_to_movie_name_mapping[new_vocab_id] = movie_name
    return movie_new_vocab_id_to_movie_name_mapping

def movie_new_vocab_id_to_movie_id():
    movie_new_vocab_id_to_movie_name_mapping = {}
    with open(MOVIE_INFO_FILE, 'r') as file:
        rows = csv.reader(file, delimiter=':')
        for row in rows:
            new_vocab_id, movie_id = row[1], row[2]
            movie_new_vocab_id_to_movie_name_mapping[new_vocab_id] = movie_id
    return movie_new_vocab_id_to_movie_name_mapping

def movie_new_vocab_id_to_movie_name_and_id():
    movie_new_vocab_id_to_movie_name_and_id_mapping = {}
    with open(MOVIE_INFO_FILE, 'r') as file:
        rows = csv.reader(file, delimiter=':')
        for row in rows:
            new_vocab_id, movie_id, movie_name = row[1], row[2], row[3]
            movie_new_vocab_id_to_movie_name_and_id_mapping[new_vocab_id] = (movie_name, movie_id)
    return movie_new_vocab_id_to_movie_name_and_id_mapping

def generate_movie_id_to_cat_id_mapping():
    movie_id_to_cat_id = collections.defaultdict(set)
    with open(MOVIE_TRAIN_FILE, 'r') as file:
        rows = csv.reader(file, delimiter=':')
        for row in rows:
            assert len(row) >= 4
            movie_id = row[1]
            cat_ids = row[2:-2]
            cat_ids = [int(c) for c in cat_ids]
            movie_id_to_cat_id[movie_id] = movie_id_to_cat_id[movie_id].union(set(cat_ids))
    return movie_id_to_cat_id