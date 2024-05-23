import os
import math
import numpy as np
import scipy.io as sio
from scipy.fftpack import fft, ifft
import scipy.signal as ss
from MyUtil import lowpass_filter, highpass_filter, butter_bandpass_filter, butter_bandpass

class EEGprocess(object):

    def __init__(self, args):

        # super(EEGprocess, self).__init__()
        self.args = args
        self.data_norm = args.data_norm
        self.bandFreqs = [
            {'name': 'Delta', 'fmin': 1,  'fmax': 3 },
            {'name': 'Theta', 'fmin': 4,  'fmax': 7 },
            {'name': 'Alpha', 'fmin': 8,  'fmax': 13},
            {'name': 'Beta ', 'fmin': 14, 'fmax': 31},
            {'name': 'Gamma', 'fmin': 31, 'fmax': 40}
        ]
        self.fs = 200
        self.window_size = self.fs
        self.window = ss.windows.hann(self.window_size)
        if self.args.dataset == "SEEDIV":
            self.label = {
                1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
            }
            self.trails = 24
            self.max_size = 64

        elif self.args.dataset == "SEED":
            self.label = {
                1: [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                2: [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                3: [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
            }
            self.trails = 15
            self.max_size = 265
        return

    def normalize(self, data):
        for channel in range(data.shape[0]):
            data[channel, :] = butter_bandpass_filter(data[channel, :], 0.3, 50, self.fs)
            mean = np.mean(data[channel, :])
            std = np.std(data[channel, :])
            data[channel, :] = (data[channel, :] - mean) / std
        return  data

    def STFT(self, epochsData, sfreq, band):
        f, t, Zxx = ss.stft(epochsData, fs=sfreq, window=self.window, nperseg=self.window_size, noverlap=None, nfft=256)
        bandResult = []
        for iter_freq in band:
            index = np.where((iter_freq['fmin'] < f) & (f < iter_freq['fmax']))
            portion = np.zeros(Zxx.shape, dtype=np.complex_)
            portion[index, :] = Zxx[index, :]
            _, xrec = ss.istft(portion, fs=sfreq)
            de = 1/2 * math.log(2 * np.pi * np.e * np.sum(pow(xrec, 2)))
            bandResult.append(de)

        return bandResult

    def de(self, data):
        de_LDS = np.zeros([data.shape[1] // self.fs, data.shape[0], 5])
        for l in range(0, data.shape[1] // 200):
            for i in range(data.shape[0]):
                de_LDS[l, i, :] = self.STFT(data[i, l*self.fs : (l+1)*self.fs], self.fs, self.bandFreqs)

        return de_LDS

    def SD_process(self, subject, session):
        skip_set = {'label.mat', 'readme.txt'}
        dir = self.args.data_path.format(dataset=self.args.dataset) + str(session) + '/'
        file = os.listdir(dir)
        subfile_path = [i for i in file if i not in skip_set if subject == int(i.split('_')[0])][0]
        feature = self.args.feature
        Train_Data, Test_Data, Train_Label, Test_Label = [], [], [], []
        # split_value = int(self.trails * 2/3)
        data = sio.loadmat(dir + subfile_path)
        session_label = self.label[session]
        test_list = []
        for i in np.unique(session_label):
            test_list.append(np.where(session_label == i)[0].tolist()[-2:])
        test_list = sum(test_list, [])

        for trail in range(1, self.trails+1):
            de_LDS = data[feature.format(trail)]
            label = session_label[trail - 1]
            if trail-1 not in test_list:
                Train_Data.append(de_LDS)
                Train_Label.append([label] * np.size(de_LDS, 1))
            else:
                Test_Data.append(de_LDS)
                Test_Label.append([label] * np.size(de_LDS, 1))

        Train_Data = np.concatenate(Train_Data, axis=1).transpose([1, 0, 2])
        Test_Data = np.concatenate(Test_Data, axis=1).transpose([1, 0, 2])
        All_Data = np.vstack((Train_Data, Test_Data))
        if self.data_norm:
            for i in range(5):
                mean_value = np.mean(Train_Data[:,:,i])
                All_Data[:,:,i] = 2 * (All_Data[:,:,i] - mean_value) / mean_value
        Train_Data = All_Data[:2010, :, :]    # 2010 X 62 X 5
        Test_Data = All_Data[2010:, :, :]
        Train_Label = np.concatenate(Train_Label, axis=0).reshape(-1, 1).astype(int)
        Test_Label = np.concatenate(Test_Label, axis=0).reshape(-1, 1).astype(int)
        adj = []
        adj_freq = []

        # adj: dot production of raw signal
        # for i in range(Train_Data.shape[2]):
        #     adj_freq = Train_Data[:,:,i].T.dot(Train_Data[:,:,i])
        #     adj_freq = (adj_freq - adj_freq.min()) / (adj_freq.max() - adj_freq.min())
        #     adj.append(adj_freq)
        # adj_freq = np.stack(adj)
        return Train_Data, Test_Data, Train_Label, Test_Label, adj_freq

    def SI_process(self, subject, session, stage):
        # skip_set = {'label.mat', 'readme.txt'}
        # dir = self.args.data_path.format(dataset=self.args.dataset) + str(session) + '/'
        # file = os.listdir(dir)
        # feature = self.args.feature
        # Train_Data, Test_Data, Train_Label, Test_Label = [], [], [], []
        # session_label = self.label[session]
        #
        # for f in file:
        #     if f not in skip_set:
        #         print(f)
        #         if subject == int(f.split('_')[0]):
        #             for trail in range(1, self.trails+1):
        #                 label = session_label[trail - 1]
        #                 de_LDS = sio.loadmat(dir + f)[feature.format(trail)]
        #                 Test_Data.append(de_LDS)
        #                 Test_Label.append([label] * np.size(de_LDS, 1))
        #         else:
        #             if stage == 'train':
        #                 for trail in range(1, self.trails + 1):
        #                     label = session_label[trail - 1]
        #                     de_LDS = sio.loadmat(dir + f)[feature.format(trail)]
        #                     Train_Data.append(de_LDS)
        #                     Train_Label.append([label] * np.size(de_LDS, 1))
        # Train_Data = np.concatenate(Train_Data, axis=1).transpose([1, 0, 2])
        # Test_Data = np.concatenate(Test_Data, axis=1).transpose([1, 0, 2])
        # All_Data = np.vstack((Train_Data, Test_Data))
        All_Data    = np.load('Data/SI_Data.npy')
        All_Label   = np.load('Data/SI_Label.npy')
        Train_Data  = All_Data[:-3394, :, :]
        Train_Label = All_Label[:-3394, :]
        Test_Label  = All_Label[-3394:, :]
        if self.data_norm:
            for i in range(5):
                mean_value = np.mean(Train_Data[:,:,i])
                All_Data[:,:,i] = 2 * (All_Data[:,:,i] - mean_value) / mean_value
        Train_Data = All_Data[:-3394 , :, :]
        Test_Data = All_Data[-3394:, :, :]     # 3394 X 62 X 5
        # Train_Label = np.concatenate(Train_Label, axis=0).reshape(-1, 1).astype(int)
        # Test_Label = np.concatenate(Test_Label, axis=0).reshape(-1, 1).astype(int)
        adj = []
        adj_freq = []

        # adj: dot production of raw signal
        # for i in range(Train_Data.shape[2]):
        #     adj_freq = Train_Data[:,:,i].T.dot(Train_Data[:,:,i])
        #     adj_freq = (adj_freq - adj_freq.min()) / (adj_freq.max() - adj_freq.min())
        #     adj.append(adj_freq)
        # adj_freq = np.stack(adj)
        return Train_Data, Test_Data, Train_Label, Test_Label, adj_freq

    def SD_freq_process(self, subject, session):
        skip_set = {'label.mat', 'readme.txt'}
        dir = './Data/SEED/Preprocessed_EEG/' + str(session) + '/'
        file = os.listdir(dir)
        subfile_path = [i for i in file if i not in skip_set if subject == int(i.split('_')[0])][0]
        Train_Data, Test_Data, Train_Label, Test_Label = [], [], [], []
        # split_value = int(self.trails * 2/3)
        data = sio.loadmat(dir + subfile_path)
        session_label = self.label[session]
        test_list = []
        for i in np.unique(session_label):
            test_list.append(np.where(session_label == i)[0].tolist()[-2:])
        test_list = sum(test_list, [])

        for trail in range(1, self.trails+1):
            eegkeys = [f for f in data.keys() if 'eeg' in f]
            eeg = data[eegkeys[trail - 1]][:, :-1]  # del one point to the integral multiple of fs
            eeg = self.normalize(eeg)
            label = session_label[trail - 1]
            de_LDS = self.de(eeg)

            if trail-1 not in test_list:
                Train_Data.append(de_LDS)
                Train_Label.append([label] * np.size(de_LDS, 1))
            else:
                Test_Data.append(de_LDS)
                Test_Label.append([label] * np.size(de_LDS, 1))

        Train_Data = np.concatenate(Train_Data, axis=1).transpose([1, 0, 2])
        Test_Data = np.concatenate(Test_Data, axis=1).transpose([1, 0, 2])
        All_Data = np.vstack((Train_Data, Test_Data))
        if self.data_norm:
            for i in range(5):
                mean_value = np.mean(Train_Data[:,:,i])
                All_Data[:,:,i] = 2 * (All_Data[:,:,i] - mean_value) / mean_value
        Train_Data = All_Data[:2010, :, :]
        Test_Data  = All_Data[2010:, :, :]
        Train_Label = np.concatenate(Train_Label, axis=0).reshape(-1, 1).astype(int)
        Test_Label  = np.concatenate(Test_Label, axis=0).reshape(-1, 1).astype(int)
        adj = []

        # adj: dot production of raw signal
        for i in range(Train_Data.shape[2]):
            adj_freq = Train_Data[:,:,i].T.dot(Train_Data[:,:,i])
            adj_freq = (adj_freq - adj_freq.min()) / (adj_freq.max() - adj_freq.min())
            adj.append(adj_freq)
        adj_freq = np.stack(adj)
        return Train_Data, Test_Data, Train_Label, Test_Label, adj_freq

    def Series_process(self, session):

        global trails
        path = []
        if session == 'all':
            for i in range(3):
                dir = self.args.data_path.format(dataset=self.args.dataset) + str(i + 1) + '/'
                file = os.listdir(dir)  # data_path means root path of data: 'Data/'
                path.append([dir + _ for _ in file])
            path = np.array(path).flatten().tolist()
        else:
            dir = self.args.data_path.format(dataset=self.args.dataset) + str(session) + '/'
            file = os.listdir(dir) # data_path means root path of data: 'Data/'
            path = [dir+_ for _ in file]
        feature = self.args.feature
        freq_num = self.args.freq_num
        Data = []
        Label = []
        Sample_num = np.zeros(len(path))

        if session == 'all':
            session_label = [self.session1_label, self.session2_label, self.session3_label]
            label = np.array(session_label).flatten()
        else:
            label = np.array(eval('session{}_label'.format(session)))


        for data_path in path:
            data = sio.loadmat(data_path)
            for j in range(trails):
                de_LDS = data[feature.format(j+1)]
                pre_pad_size = int(np.floor((self.max_size - np.size(de_LDS, 1)) /2))
                post_psd_size = self.max_size - pre_pad_size - np.size(de_LDS, 1)
                de_LDS = np.pad(de_LDS, ((0, 0), (pre_pad_size, post_psd_size), (0, 0)), 'constant',
                                constant_values = 0).reshape(62, self.max_size * freq_num)
                de_LDS = np.expand_dims(de_LDS, axis = 0)
                Data.append(de_LDS)
                Label.append(label[j])
                Sample_num[i] = trails

        Data = np.concatenate(Data, axis=0)
        Label = np.array(Label).reshape(-1, 1).astype(int)
        Sample_num = Sample_num.reshape(-1, 1).astype(int)



        return Data, Label, Sample_num

    def Sample_process(self, session):

        global trails
        dir = self.args.data_path.format(dataset=self.args.dataset) + str(session) + '/'
        file = os.listdir(dir)
        feature = self.args.feature
        Data = []
        Label = []
        Sample_Num = np.zeros(len(file))

        label = np.array(self.label[session])

        for i in range(len(file)):
            data = sio.loadmat(dir + file[i])
            if i < len(file):
                for j in range(trails):
                    de_LDS = data[feature.format(j+1)]
                    Data.append(de_LDS)
                    Label.append([label[j]] * np.size(de_LDS, 1))
                    Sample_Num[i] += np.size(de_LDS, 1)

        Data  = np.concatenate(Data,  axis = 1).transpose([1, 0, 2])
        Label = np.concatenate(Label, axis = 0).reshape(-1, 1).astype(int)
        Sample_Num = Sample_Num.reshape(-1, 1).astype(int)


        return Data, Label, Sample_Num

# if  __name__== "__main__":
#
#     session = 1
#     dir = '../Data/' + str(session) + '/'
#     feature = 'de_LDS{}'
#
#     max_size = 64
#     channel_num = 5
#
#     # Data, Label = Series_process(dir, feature, session, max_size, channel_num)
#     # Data, Label, Sample_Num= Sample_process(dir, feature, session, max_size, channel_num)
#     #
#     # np.save( 'Data/Train_data{}'.format(session) + '.npy', Data)
#     # np.save('Data/Train_label{}'.format(session) + '.npy', Label)
