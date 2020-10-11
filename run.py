import re
import os
import csv
import math
import torch
import numpy as np
import torch.nn.functional as F

from model.two_tower import TwoTower
import metrics.recall_k as metrics

EPOCHS = 15
BATCH_SIZE = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_meta_info(meta_data):
    n_user, n_movie, n_category = 0, 0, 0
    with open(meta_data, newline='') as file:
        rows = csv.reader(file, delimiter=':', quotechar='|')
        for row in rows:
            if row[0] == 'num_user':
                n_user = int(row[1])
            if row[0] == 'num_movie':
                n_movie = int(row[1])
            if row[0] == 'num_category':
                n_category = int(row[1])
    return n_user, n_movie, n_category


def train_test_model(model, device, train_data, test_data, n_category, k):
    """
    train and test the model # of epochs
    :param model: The NN based model
    :param train_data: where the train data locates
    :param test_data: where the test data locates
    :param n_category: total number of categories 
    :param k: this is for test model recall@k 
    :return:
    """
    print("Start Training Model!")

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS + 1):
        train_model(model, train_data, device, optimizer, epoch, n_category)
    
    test_model(model, test_data, device, n_category, k)


def train_model(model, train_data, device, optimizer, epoch, n_category):
    """
    train one epoch of the model batch by batch
    :param model: The NN based model
    :param train_data: where the train data locates
    :param optimizer: use optimizer for update weights
    :param epoch: the current epoch
    :param n_category: total number of categories 
    :return:
    """
    train_item_count = count_in_file_items(train_data)

    user_idx, movie_idx, category_idx_lst, labels = get_idx_label(train_data, n_category)
    
    # train the model per batcb
    for batch_idx in range(math.ceil(train_item_count / BATCH_SIZE)):
        # get the start and end index for the current batch
        st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
        ed_idx = min(ed_idx, train_item_count - 1)

        batch_user_idx = user_idx[st_idx:ed_idx]
        batch_movie_idx = movie_idx[st_idx:ed_idx]
        batch_category_idx_lst = category_idx_lst[st_idx:ed_idx, :]
        batch_labels = labels[st_idx:ed_idx]

        # convert the numpy format to torch tensor format
        u_idx = torch.LongTensor([int(x) for x in batch_user_idx])
        u_idx = u_idx.to(device)
        m_idx = torch.LongTensor([int(x) for x in batch_movie_idx])
        m_idx = m_idx.to(device)
        c_idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_category_idx_lst])
        c_idx = c_idx.to(device)

        batch_labels = torch.from_numpy(batch_labels)
        target = batch_labels.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        # 1. FORWARD PASS
        output = model(u_idx, m_idx, c_idx)
        loss = torch.nn.MSELoss()(output, target)

        # 2. BACKWARD PASS/BACK PROPOGATION
        loss.backward()
        
        # 3. Update
        optimizer.step()
        
        print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss:{:.6f}'.format(
            epoch, batch_idx * len(u_idx), train_item_count,
            100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.item()))


def test_model(model, test_data, device, n_category, k):
    """
    test the model
    :param model: The NN based model
    :param test_data: where the test data locates
    :param n_category: total number of categories 
    :param k: this is for test model recall@k
    :return:
    """
    user_idx, movie_idx, category_idx_lst, labels = get_idx_label(test_data, n_category)
    test_loss = 0
    recall_k = [0] * len(k)
    test_item_count = count_in_file_items(test_data)
    with torch.no_grad():
        for batch_idx in range(math.ceil(test_item_count / BATCH_SIZE)):
            st_idx, ed_idx = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
            ed_idx = min(ed_idx, test_item_count - 1)

            batch_user_idx = user_idx[st_idx:ed_idx]
            batch_movie_idx = movie_idx[st_idx:ed_idx]
            batch_category_idx_lst = category_idx_lst[st_idx:ed_idx, :]
            batch_labels = labels[st_idx:ed_idx]

            u_idx = torch.LongTensor([int(x) for x in batch_user_idx]).to(device)
            m_idx = torch.LongTensor([int(x) for x in batch_movie_idx]).to(device)
            c_idx = torch.LongTensor([[int(x) for x in x_idx] for x_idx in batch_category_idx_lst]).to(device)

            target = torch.from_numpy(batch_labels).to(device, dtype=torch.float32)
           
            # 1. Forward Pass (Predict does not need BP and update)
            output = model(u_idx, m_idx, c_idx)
            test_loss += torch.nn.MSELoss()(output, target)

            # TODO: calculate recall@k for all data. 
            # recall@k is slow, here we only count first batch.
            if batch_idx == 0:
                for k_idx, kk in enumerate(k):
                    recall_k[k_idx] += metrics.augmented_recallK_batch_users(model, u_idx, kk)

        test_loss /= math.ceil(test_item_count / BATCH_SIZE)
        print('Test set: Average MSE loss: {:.5f}'.format(test_loss))
        print('Test set: Average recall@{}: {:.5f}'.format(kk, recall_k[k_idx]))
        

def count_in_file_items(fname):
    """
    count the number of lines of a file
    """
    count = 0
    with open(fname.strip(), 'r') as fin:
        for _ in fin:
            count += 1
    return count


def get_idx_label(fname, n_category, shuffle=True):
    """
    read the train dataset from file, and return idx and value for each training example
    :param fname: file name
    :param n_category: total number of categories
    :param shuffle: shuffle the training examples
    :return: idx and value for all the training examples
    """
    # process one training example
    def _process_line(line, n_category):
        features = line.rstrip('\n').split(':')
        assert len(features) >= 3, "There exists missing features, check the training example."
        u_idx = [int(features[0])]
        m_idx = [int(features[1])]
        # TODO: fix c_idx 
        c_idx = [int(c) for c in features[2:-2]] + (n_category-len(features[2:-2]))*[-1]
        label = int(features[-2])/10.0

        return u_idx, m_idx, c_idx, label

    user_idxs, movie_idxs, category_idxs, labels = [], [], [], []
    with open(fname.strip(), 'r') as fin:
        for line in fin:
            user_idx, movie_idx, category_idx, label = _process_line(line, n_category)
            user_idxs.append(user_idx)
            movie_idxs.append(movie_idx)
            category_idxs.append(category_idx)
            labels.append(label)

    user_idxs = np.array(user_idxs)
    movie_idxs = np.array(movie_idxs)
    category_idxs = np.array(category_idxs)
    labels = np.array(labels)

    # shuffle
    if shuffle:
        idx_list = np.arange(len(user_idxs))
        np.random.shuffle(idx_list)

        user_idxs = user_idxs[idx_list, :]
        movie_idxs = movie_idxs[idx_list, :]
        category_idxs = category_idxs[idx_list, :]
        labels = labels[idx_list]

    return user_idxs, movie_idxs, category_idxs, labels

def run():
    # specify dataset and the hyper-parameters
    train_data = "data/train_data_sorted_vocab_id.dat" 
    test_data = "data/test_data_sorted_vocab_id.dat"
    n_user, n_movie, n_category = get_meta_info('data/train_meta_data.dat')
    apply_qat = False
    lr = 0.012
    popularity_hashing = True
    pHash = 0.7
    recall_k = [5, 10]

    # define the model
    model = TwoTower(num_user=n_user, num_movie=n_movie, 
                            num_category=n_category, popularity_hashing=popularity_hashing,
                            uid_ph=[pHash, 0.9], mid_ph=[pHash, 0.9]).to(DEVICE)
    if apply_qat:
        model.qconfig=torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)

    # train the model
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    for epoch in range(1, EPOCHS + 1):
        train_model(model, train_data, DEVICE, optimizer, epoch, n_category)

    # save the model for inference
    if popularity_hashing:
        save_file = "checkpoints/ep_" + str(EPOCHS) + "_lr_" + str(lr) + "_pHash_" + str(pHash)
    else:
        save_file = "checkpoints/ep_" + str(EPOCHS) + "_lr_" + str(lr) + "baseline" 
    if apply_qat:
        model=torch.quantization.convert(model.eval(), inplace=True)
        torch.save(model.state_dict(), save_file + "_qat")
    else:
        torch.save(model.state_dict(), save_file)
    
    # test the model
    test_model(model, test_data, DEVICE, n_category, recall_k)


if __name__ == '__main__':
    run()

    