import sys
sys.path.append("./../")
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as dataf
from scipy.io import loadmat
import time
from torchsummary import summary
import torch.backends.cudnn as cudnn
import utils
import logger
from MuSeS_Net import MuSeS
cudnn.deterministic = True
cudnn.benchmark = False

class HSI_LiDAR_DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, dataset='MUUFL'):
        HSI = loadmat(f'./../{dataset}11x11/HSI_Tr.mat')
        LiDAR = loadmat(f'./../{dataset}11x11/LIDAR_Tr.mat')
        label = loadmat(f'./../{dataset}11x11/TrLabel.mat')
        self.hs_image = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0, 3, 1, 2)
        self.lidar_image = (torch.from_numpy(LiDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0, 3, 1, 2)
        self.lbls = ((torch.from_numpy(label['Data']) - 1).long()).reshape(-1)
    def __len__(self):
        return self.hs_image.shape[0]
    def __getitem__(self, i):
        return self.hs_image[i], self.lidar_image[i], self.lbls[i]

class HSI_LiDAR_DatasetTest(torch.utils.data.Dataset):
    def __init__(self, dataset='MUUFL'):
        HSI = loadmat(f'./../{dataset}11x11/HSI_Te.mat')
        LiDAR = loadmat(f'./../{dataset}11x11/LIDAR_Te.mat')
        label = loadmat(f'./../{dataset}11x11/TeLabel.mat')
        self.hs_image = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0, 3, 1, 2)
        self.lidar_image = (torch.from_numpy(LiDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0, 3, 1, 2)
        self.lbls = ((torch.from_numpy(label['Data']) - 1).long()).reshape(-1)
    def __len__(self):
        return self.hs_image.shape[0]
    def __getitem__(self, i):
        return self.hs_image[i], self.lidar_image[i], self.lbls[i]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
datasetNames = ["MUUFL","Trento", "Houston"]
MultiModalData = 'LiDAR'
modelName = 'MuSeS-Net'
patchsize = 11
batch_size = 64
test_batch_size = 500
EPOCHS = 100
learning_rate = 5e-4
FM = 16
FileName = modelName
num_iterations = 10
train_losses = []
test_losses = []

def seed_val(seed=14):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for dataset in datasetNames:
    print(f"---------------------------------- Details for {dataset} dataset---------------------------------------------")
    print('\n')
    try:
        os.makedirs(dataset)
    except FileExistsError:
        pass

    train_dataset = HSI_LiDAR_DatasetTrain(dataset=dataset)
    test_dataset = HSI_LiDAR_DatasetTest(dataset=dataset)
    NC = train_dataset.hs_image.shape[1]
    NCLidar = train_dataset.lidar_image.shape[1]
    Classes = len(torch.unique(train_dataset.lbls))
    train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_patch_hsi = test_dataset.hs_image
    test_patch_lidar = test_dataset.lidar_image
    test_label = test_dataset.lbls

    KAPPA = []
    OA = []
    AA = []
    ELEMENT_ACC = np.zeros((num_iterations, Classes))

    seed_val(14)
    for iterNum in range(num_iterations):
        print('\n')
        print("---------------------------------- Summary ---------------------------------------------")
        print('\n')
        model = MuSeS(in_channels=NC,in_channels_fused=NC+NCLidar, h=patchsize, w=patchsize, class_count=Classes).cuda()
        summary(model, [(NC, patchsize, patchsize), (NCLidar, patchsize, patchsize)], device='cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train_loss_func = nn.CrossEntropyLoss()
        test_loss_func = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        BestAcc = 0
        torch.cuda.synchronize()
        print('\n')
        print(f"---------------------------------- Training started for {dataset} dataset ---------------------------------------------")
        print('\n')
        start = time.time()
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0

            for step, (batch_hsi, batch_ldr, batch_lbl) in enumerate(train_loader):
                batch_hsi = batch_hsi.cuda()
                batch_ldr = batch_ldr.cuda()
                batch_lbl = batch_lbl.cuda()
                out = model(batch_hsi, batch_ldr)
                loss = train_loss_func(out, batch_lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            model.eval()
            epoch_test_loss = 0
            y_pred = np.empty((len(test_label)), dtype='float32')
            number = len(test_label) // test_batch_size
            with torch.no_grad():
                for i in range(number):
                    temp = test_patch_hsi[i * test_batch_size:(i + 1) * test_batch_size, :, :]
                    temp = temp.cuda()
                    temp1 = test_patch_lidar[i * test_batch_size:(i + 1) * test_batch_size, :, :]
                    temp1 = temp1.cuda()
                    temp2 = model(temp, temp1)
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    y_pred[i * test_batch_size:(i + 1) * test_batch_size] = temp3.cpu()
                    epoch_test_loss += test_loss_func(temp2, test_label[i * test_batch_size:(i + 1) * test_batch_size].cuda()).item()
                    del temp, temp1, temp2, temp3

                if (i + 1) * test_batch_size < len(test_label):
                    temp = test_patch_hsi[(i + 1) * test_batch_size:len(test_label), :, :]
                    temp = temp.cuda()
                    temp1 = test_patch_lidar[(i + 1) * test_batch_size:len(test_label), :, :]
                    temp1 = temp1.cuda()
                    temp2 = model(temp, temp1)
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    y_pred[(i + 1) * test_batch_size:len(test_label)] = temp3.cpu()
                    epoch_test_loss += test_loss_func(temp2, test_label[(i + 1) * test_batch_size:len(test_label)].cuda()).item()
                    del temp, temp1, temp2, temp3

                epoch_test_loss /= len(test_label) // test_batch_size + 1
                test_losses.append(epoch_test_loss)

                y_pred = torch.from_numpy(y_pred).long()
                accuracy = torch.sum(y_pred == test_label).type(torch.FloatTensor) / test_label.size(0)

                print('Epoch: ', epoch, '| train loss: %.4f' % epoch_train_loss, '| test loss: %.4f' % epoch_test_loss, '| test accuracy: %.4f' % (accuracy * 100))

                if accuracy > BestAcc:
                    BestAcc = accuracy
                    torch.save(model.state_dict(), dataset + '/net_params_' + FileName + '.pkl')

            scheduler.step()
        torch.cuda.synchronize()
        end = time.time()
        print('\nThe train time (in seconds) is:', end - start)
        Train_time = end - start
        model.load_state_dict(torch.load(dataset + '/net_params_' + FileName + '.pkl'))
        model.eval()
        confusion_mat, overall_acc, class_acc, avg_acc, kappa_score = utils.result_reports(test_patch_hsi, test_patch_lidar, test_label, dataset, model, modelName, iterNum)
        KAPPA.append(kappa_score)
        OA.append(overall_acc)
        AA.append(avg_acc)
        ELEMENT_ACC[iterNum, :] = class_acc
        torch.save(model, dataset + '/best_model_' + FileName + '_Iter' + str(iterNum) + '.pt')
        print('\n')
        print("Overall Accuracy = ", overall_acc)
        print('\n')
    print(f"---------- Training Finished for {dataset} dataset -----------")
    print("\nThe Confusion Matrix")
    logger.log_result(OA, AA, KAPPA, ELEMENT_ACC, './' + dataset + '/' + FileName + '_Report_' + dataset + '.txt')
