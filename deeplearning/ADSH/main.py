import pickle
import os
import argparse
import logging
import torch
import time
import json

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.data_processing as dp
import utils.adsh_loss as al
import utils.cnn_model as cnn_model
import utils.subset_sampler as subsetsampler
import utils.calc_hr as calc_hr

# ******************* need modification ******************
# *********************************************************
# need tuning: bits, max_iter, epochs, batch_size, num_samples
parser = argparse.ArgumentParser(description="ADSH demo")
parser.add_argument('--bits', default='12,24,32,48', type=str,
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 1)')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=50, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=3, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size (default: 64)')

parser.add_argument('--num-samples', default=1000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--gamma', default=200, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
# *********************************************************
# *********************************************************


def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return


def _save_npz(rB, qB):
    hashcode = np.concatenate((rB, qB), axis=0)
    np.savez('./hashcode_{}.npz'.format(parser.parse_args().bits), ans=hashcode)


def _save_json():
    indice = train_index + test_index
    image_names = [file_names[i] for i in indice]
    hash_json = {'name': image_names}
    with open('./hash_{}_json.json'.format(parser.parse_args().bits), 'w', encoding='utf-8') as json_file:
        json.dump(hash_json, json_file)
        print("write json file success!")


def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return


def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


# ************************* need modification *********************
# *****************************************************************
def _dataset():
    """Return the data.
    Returns: a tuple in the form of (nums, dsets, labels),
        where dsets = (dset_database, dset_test)
              nums = (num_database, num_test)
              labels = (databaselabels, testlabels),
        where dset_database, dset_test: torch.utils.data.dataset, training and test sets;
              num_database, num_test: int, size of training and test sets;
              databaselabels, testlabels: labels of training and test sets.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # load training and test data sets
    data_path = r'./images'
    global file_names, train_index, test_index
    file_names, labels = [], []
    with open('./file_names.txt', 'r') as f:
        for line in f:
            file_names.append(line.strip())
    with open('./labels.txt', 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    train_index, test_index = train_test_split(range(len(file_names)), test_size=0.2, random_state=42, stratify=labels)
    # random_list = list(np.random.permutation(range(len(file_list))))
    # train_index, test_index = random_list[: int(0.9*len(file_list))], random_list[int(0.9*len(file_list)): ]
    # train_files = [file_list[i] for i in train_index]
    # test_files = [file_list[i] for i in test_index]
    dset_database = dp.DatasetProcessing(
        data_path, 'file_names.txt', 'labels.txt', train_index,  transformations)
    dset_test = dp.DatasetProcessing(
        data_path, 'file_names.txt', 'labels.txt', test_index,  transformations)
    num_database, num_test = len(dset_database), len(dset_test)

    # load corresponding labels
    def load_label(labels, indice):
        return torch.LongTensor(list(map(int, [labels[i] for i in indice])))

    databaselabels = load_label(labels, train_index)
    testlabels = load_label(labels, test_index)

    nclasses = 37
    databaselabels = encoding_onehot(databaselabels, nclasses=nclasses)
    testlabels = encoding_onehot(testlabels, nclasses=nclasses)

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels
# *****************************************************************
# *****************************************************************


def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S


def calc_loss(V, U, S, code_length, select_index, gamma):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length * S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (U - V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)
    return loss


def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B


def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10


def adsh_algo(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    gamma = opt.gamma

    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)
    model.cuda()
    adsh_loss = al.ADSHLoss(gamma, code_length, num_database)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    V = np.zeros((num_database, code_length))

    model.train()
    for iter in range(max_iter):
        iter_time = time.time()
        '''
        sampling and construct similarity matrix
        '''
        select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 sampler=_sampler,
                                 drop_last=True,
                                 shuffle=False,
                                 num_workers=4)
        '''
        learning deep neural network: feature learning
        '''
        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, database_labels)
        U = np.zeros((num_samples, code_length), dtype=np.float)
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration + 1) * batch_size)) - 1,
                                    batch_size_, dtype=int)
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                S = Sim.index_select(0, torch.from_numpy(u_ind))
                U[u_ind, :] = output.cpu().data.numpy()

                model.zero_grad()
                loss = adsh_loss(output, V, S, V[batch_ind.cpu().numpy(), :])
                loss.backward()
                optimizer.step()
        adjusting_learning_rate(optimizer, iter)

        '''
        learning binary codes: discrete coding
        '''
        barU = np.zeros((num_database, code_length))
        barU[select_index, :] = U
        Q = -2 * code_length * Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
        iter_time = time.time() - iter_time
        loss_ = calc_loss(V, U, Sim.cpu().numpy(), code_length, select_index, gamma)
        logger.info('[Iteration: %3d/%3d][Train Loss: %.4f]', iter, max_iter, loss_)
        record['train loss'].append(loss_)
        record['iter time'].append(iter_time)

    '''
    training procedure finishes, evaluation
    '''
    model.eval()
    testloader = DataLoader(dset_test, batch_size=1,
                            drop_last=True,
                            shuffle=False,
                            num_workers=4)
    qB = encode(model, testloader, num_test, code_length)
    rB = V
    map = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
    logger.info('[Evaluation: mAP: %.4f]', map)
    record['rB'] = rB
    record['qB'] = qB
    record['map'] = map
    filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)
    _save_json()


if __name__ == "__main__":
    global opt, logdir
    opt = parser.parse_args()
    # ************************** need modification ********************
    # *****************************************************************
    # set the path of the log
    logdir = '-'.join([r'./log/log-ADSH-cifar10', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    # *****************************************************************
    # *****************************************************************
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        adsh_algo(bit)