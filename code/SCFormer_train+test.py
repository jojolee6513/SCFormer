import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math
import argparse
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from model.SCFormer_model import MultiScaleMaskedTransformerDecoder as mynet
import time
import utils

# np.random.seed(1337)

torch.cuda.set_device(0)
class_num = 9
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim", type=int, default=160)
parser.add_argument("-c","--src_input_dim", type=int, default=128)
parser.add_argument("-d","--tar_input_dim", type=int, default=103)  # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default=64)
parser.add_argument("-w","--class_num",type = int, default=class_num)  #UP class
parser.add_argument("-s","--shot_num_per_class", type=int, default = 1)
parser.add_argument("-b","--query_num_per_class", type=int, default = 19)
parser.add_argument("-e","--episode", type=int, default=20000)
parser.add_argument("-t","--test_episode", type=int, default=600)
parser.add_argument("-l","--learning_rate", type=float, default=0.0005)  #0.0005 0.001
# parser.add_argument("-g","--gpu",type=int, default=1)
parser.add_argument("-u","--hidden_unit", type=int, default=10)
# target
parser.add_argument("-m","--test_class_num", type=int, default=class_num)
parser.add_argument("-z","--test_lsample_num_per_class", type=int, default=4, help='5 4 3 2 1')
args = parser.parse_args(args=[])

# Hyper Parameters
Batch_size = 64
PATCHSIZE_half = 3
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
# GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
TEST_CLASS_NUM = args.test_class_num  # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class  # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)
res_name = 'SCFormer_UP_'
checkpoints_path = 'ckpt'
if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

classificationMap_path = './classificationMap/'
# if not os.path.exists(classificationMap_path):
#     os.makedirs(classificationMap_path)

# load source domain data set
with open(os.path.join('/home/data/zhangzhiyuan/Chikusei/', 'Chikusei_imdb_128_7.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']  # (77592, 9, 9, 128)
labels_train = source_imdb['Labels']  # 77592
# patchlab_train = source_imdb['patch_label']
print(data_train.shape)
print(labels_train.shape)
# print(patchlab_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}  
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data)  # 200 labels samples per class,class 18 only have 145 samples, which has been delete
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,128）-> (128,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data  # 18 classes, 200samples/class
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape)  # (77592, 9, 9, 128)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))  #(9, 9, 100, 77592)
print(source_imdb['data'].shape)  #(9, 9, 100, 77592)
print(source_imdb['Labels'].shape)
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=Batch_size, shuffle=True, num_workers=0, drop_last=True)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
test_data = '/home/data/zhangzhiyuan/PaviaUniversity/paviaU.mat'
test_label = '/home/data/zhangzhiyuan/PaviaUniversity/paviaU_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data2(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)  # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = PATCHSIZE_half  # 4 patchsize
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS   # 5
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)  # 返回数字的上入整数。

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0, drop_last=True)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=Batch_size, shuffle=True, num_workers=0, drop_last=True)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain


crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


# run 10 times
nDataSet = 2
EPISODE = 100  # 10000
MSE_max = 100000
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_episdoe_record = [i for i in range(10)]
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

seeds = [1330, 1320, 2346, 1320, 1224, 1236, 1226, 1235, 1233, 1229]  # 2 3


# train
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])

    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=class_num, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    # model
    network = mynet(
        in_channels=64,
        mask_classification=True,
        num_classes=class_num,
        hidden_dim=49,
        num_queries=1,
        nheads=7,
        dim_feedforward=49 * 4,
        dec_layers=4,
        pre_norm=True,
    )

    network.load_state_dict(torch.load("/home/zhangzhiyuan/code/IEEE_TIP_2024_SCFormer/model/UP_pre.pkl"))
    # network.load_state_dict(torch.load("/home/code/IEEE_TIP_2024_SCFormer/model/SA_pre.pkl"))
    # network.load_state_dict(torch.load("/home/code/IEEE_TIP_2024_SCFormer/model/IP_pre.pkl"))
    # network.load_state_dict(torch.load("/home/code/IEEE_TIP_2024_SCFormer/model/PC_pre.pkl"))

    network.cuda()
    network.train()
    network_optim = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []

    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(EPISODE):  # EPISODE = 90000
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()

        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_outputs, query_outputs, intra_domain_loss = network(x=supports.cuda(), y=querys.cuda(),
                                                                        y_label=query_labels.cuda(),
                                                                        domain='source')
            wd_loss = network(wd_s=source_data.cuda(), wd_t=target_data.cuda())
            wd_loss = wd_loss[0]
            # FSL loss
            logits = euclidean_metric(query_outputs, support_outputs)
            cls_loss = crossEntropy(logits, query_labels.cuda())
            loss = cls_loss + 0.001 * wd_loss + 0.1 * intra_domain_loss ############# OA: 75.06

            # Update parameters
            network.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.1, norm_type=2)
            network_optim.step()

            episode_correct_num = torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_hit += episode_correct_num
            total_num += querys.shape[0]

            print('episode {:>3d}:  intra-domain loss: {:6.4f}, inter-domain loss: {:6.4f}, '
                  'cls loss: {:6.4f}, eposide acc: {:6.4f}, iteration acc {:6.4f}, loss: {:6.4f}'.format(
                episode + 1, intra_domain_loss.item(), wd_loss.item(),
                cls_loss.item(), episode_correct_num / querys.shape[0], total_hit / total_num, loss.item()))

        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, class_num, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            # calculate features
            support_outputs, query_outputs, intra_domain_loss = network(x=supports.cuda(), y=querys.cuda(),
                                                                        y_label=query_labels.cuda(),
                                                                        domain='target')
            wd_loss = network(wd_s=source_data.cuda(), wd_t=target_data.cuda())
            wd_loss = wd_loss[0]

            # FSL loss
            logits = euclidean_metric(query_outputs, support_outputs)
            cls_loss = crossEntropy(logits, query_labels.cuda())

            loss = cls_loss + 0.001*wd_loss + 0.1*intra_domain_loss ############# OA: 76.84

            # Update parameters
            network.zero_grad()
            loss.backward()
            network_optim.step()

            episode_correct_num = torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_hit += episode_correct_num
            total_num += querys.shape[0]

            print('episode {:>3d}:  intra-domain loss: {:6.4f}, inter-domain loss: {:6.4f}, '
                  'cls loss: {:6.4f}, eposide acc: {:6.4f}, iteration acc: {:6.4f}, loss: {:6.4f}'.format(
                episode + 1, intra_domain_loss.item(), wd_loss.item(),
                cls_loss.item(), episode_correct_num / querys.shape[0], total_hit / total_num, loss.item()))


        # #test
        #
        if (episode + 1) % 100 == 0 or episode == 0:
            # test
            print("Testing ...")

            network.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()
            train_features = network(Variable(train_datas).cuda(), domain='target')  # (45, 160)

            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
            train_end = time.time()
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features = network(Variable(test_datas).cuda(), domain='target')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()
            print('test time: ', test_end - train_end)

            # Training mode
            network.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(network.state_dict(), checkpoints_path + str("/" + res_name + str(iDataSet) + "iter_" + str(
                    TEST_LSAMPLE_NUM_PER_CLASS) + "shot_" + str(test_accuracy) + ".pkl"))
                print("save networks for episode:", episode + 1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
                best_episdoe_record[iDataSet] = best_episdoe + 1
            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

################################################################################################################
    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')


# # ac
AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

print('best episode record: {}'.format(best_episdoe_record))

#################classification map################################

for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
label = best_G

for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        if label[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if label[i][j] == 1:
            hsi_pic[i, j, :] = [230, 25, 75]
        if label[i][j] == 2:
            hsi_pic[i, j, :] = [60, 180, 75]
        if label[i][j] == 3:
            hsi_pic[i, j, :] = [255, 255, 25]
        if label[i][j] == 4:
            hsi_pic[i, j, :] = [67, 99, 216]
        if label[i][j] == 5:
            hsi_pic[i, j, :] = [245, 130, 49]
        if label[i][j] == 6:
            hsi_pic[i, j, :] = [145, 30, 180]
        if label[i][j] == 7:
            hsi_pic[i, j, :] = [66, 212, 244]
        if label[i][j] == 8:
            hsi_pic[i, j, :] = [240, 50, 230]
        if label[i][j] == 9:
            hsi_pic[i, j, :] = [191, 239, 69]
        if label[i][j] == 10:
            hsi_pic[i, j, :] = [250, 190, 212]
        if label[i][j] == 11:
            hsi_pic[i, j, :] = [70, 153, 144]
        if label[i][j] == 12:
            hsi_pic[i, j, :] = [220, 190, 255]
        if label[i][j] == 13:
            hsi_pic[i, j, :] = [154, 99, 36]
        if label[i][j] == 14:
            hsi_pic[i, j, :] = [255, 250, 200]
        if label[i][j] == 15:
            hsi_pic[i, j, :] = [128, 0, 0]
        if label[i][j] == 16:
            hsi_pic[i, j, :] = [170, 255, 195]
        if label[i][j] == 17:
            hsi_pic[i, j, :] = [128, 128, 0]
        if label[i][j] == 18:
            hsi_pic[i, j, :] = [255, 216, 177]
        if label[i][j] == 19:
            hsi_pic[i, j, :] = [0, 0, 117]
import cv2
path_final = classificationMap_path+res_name + '.png'
# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24, "./0923_classificationMap/Maskformer_UP{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
# classification_map(hsi_pic[4:-4, 4:-4, :], label[4:-4, 4:-4], 24, path_final)
hsi_pic = hsi_pic.astype(np.uint8)
hsi_pic = cv2.cvtColor(hsi_pic, cv2.COLOR_RGB2BGR)
cv2.imwrite(path_final, hsi_pic)
