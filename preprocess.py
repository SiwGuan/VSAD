from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
import numpy as np
import zipfile
import numpy as np
import pandas

from args import get_parser
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


def convertNumpytrain(df):
    x = df[df.columns[3:]].values[::1, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)
def convertNumpy(df):
    x = df[df.columns[3:]].values[::1, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    np.save(path.join(output_folder, dataset + "_" + category + ".npy"), temp)


def load_data(dataset):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "processed/ServerMachineDataset"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )

    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/SMAP_MSL"
        if dataset == "SMAP":
            output_folder = "processed/SMAP/"
        if dataset == "MSL":
            output_folder = "processed/MSL/"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0]: anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            np.save(output_folder + category + '.npy', data)

        for c in ["train", "test"]:
            concatenate_and_save(c)
        print(dataset, "test_label", labels.shape)
        labels = np.array(labels, dtype=np.float32)
        np.save(output_folder + 'labels.npy', labels)
    elif dataset == "SWAT":
        z_tr = zipfile.ZipFile("./datasets/SWaT/SWaT_train.zip", "r")
        f_tr = z_tr.open(z_tr.namelist()[0])
        train = pd.read_csv(f_tr)
        f_tr.close()
        z_tr.close()
        class_map = {"Normal": 0, "Attack": 1}
        train["Normal/Attack"] = train["Normal/Attack"].map(class_map)
        train["Timestamp"] = 0
        train = np.array(train, dtype=np.float64)
        train_data = train[:, 1:-1]
        train_label = train[:, -1]

        z_tr = zipfile.ZipFile("./datasets/SWaT/SWaT_test.zip", "r")
        f_tr = z_tr.open(z_tr.namelist()[0])
        test = pd.read_csv(f_tr)
        f_tr.close()
        z_tr.close()
        test["Timestamp"] = 0
        class_map = {"Normal": 0, "Attack": 1}
        test["Normal/Attack"] = test["Normal/Attack"].map(class_map)
        test = np.array(test, dtype=np.float64)
        test_data = test[:, 1:-2]
        test_label = test[:, -1]

        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        print("test set label shape: ", None if test_label is None else test_label.shape)
        np.save('./processed/SWAT/train.npy', train_data)
        np.save('./processed/SWAT/test.npy', test_data)
        np.save('./processed/SWAT/labels.npy', test_label)
    elif dataset == "WADI_init":
        z_tr = zipfile.ZipFile("./datasets/WADI/WADI_train.zip", "r")
        f_tr = z_tr.open(z_tr.namelist()[0])
        train = pd.read_csv(f_tr)
        f_tr.close()
        z_tr.close()
        train_data = np.array(train, dtype=np.float64)

        z_tr = zipfile.ZipFile("./datasets/WADI/WADI_test.zip", "r")
        f_tr = z_tr.open(z_tr.namelist()[0])
        test = pd.read_csv(f_tr)
        f_tr.close()
        z_tr.close()
        test["Time"] = 0
        test = np.array(test, dtype=np.float64)
        test_data = test[:, 1:-1]
        test_label = test[:, -1]
        # test_label = np.tile(test_label, (test_data.shape[1], 1)).transpose(1, 0)   #设置所有特征维度的标签

        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        print("test set label shape: ", None if test_label is None else test_label.shape)

        np.save('./processed/WADI/train.npy', train_data)
        np.save('./processed/WADI/test.npy', test_data)
        np.save('./processed/WADI/labels.npy', test_label)
    elif dataset == 'WADI':
        dataset_folder = './datasets/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=100, nrows=3e6, header=None)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True);
        test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True);
        test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i);
                        break
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpytrain(train), convertNumpy(test), convertNumpy(labels)
        labels = np.array(np.array(np.sum(labels, axis=1), dtype=bool), dtype=np.float32)
        print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join("./processed/WADI", f'{file}.npy'), eval(file))
    elif dataset == "SYNTHETIC":
        train_file = os.path.join("datasets", dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join("datasets", dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000

        def normalize(a):
            a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
            return (a / 2 + 0.5)

        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros(test.shape)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30, lab.values[i][1:]] = 1
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join('./processed/synthetic', f'{file}.npy'), eval(file))
    elif dataset == "PSM":
        train_data = pandas.read_csv("./datasets/PSM/train.csv")
        test_data = pandas.read_csv("./datasets/PSM/test.csv")
        labels = pandas.read_csv("./datasets/PSM/test_label.csv")
        labels = labels["label"]
        test_label = np.array(labels)
        train_data = np.array(train_data, dtype=np.float32)[:, 1:]
        test_data = np.array(test_data, dtype=np.float32)[:, 1:]
        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        print("test set label shape: ", None if test_label is None else test_label.shape)

        np.save('./processed/PSM/train.npy', train_data)
        np.save('./processed/PSM/test.npy', test_data)
        np.save('./processed/PSM/labels.npy', test_label)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)
