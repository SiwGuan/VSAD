import numpy as np
import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
from eval_methods import *
from diagnosis import *
from settings import *
import os
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
import json
import time
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
class Trainer():
    def __init__(self, model, dataset, lr, weight_decay, device, clip, train_loader, val_loader, test_loader,gamma,theta,x_train,x_test,n_epochs,window_size,
                 n_features,save_path,summary_file_name="summary.txt"):
        self.model = model
        self.model.to(device)
        self.loss = nn.MSELoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip = clip
        self.theta = theta
        self.n_epochs = n_epochs
        self.device = device
        self.dataset = dataset
        self.num_nodes = n_features
        self.num_split = 1
        self.window_size = window_size
        self.traindata = x_train
        self.testdata = x_test
        self.schedular = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=gamma)
        self.losses = {
            "train": [],
            "val": [],
            "train_scores": [],
            "test_scores": []
        }
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_path = save_path
        self.summary_file_name = summary_file_name
        self.val_min_loss = None
    def train(self):
        start_time = time.time()
        for epoch in range(self.n_epochs):
            self.model.train()
            for iter, (x, y) in enumerate(tqdm(self.train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                if x.ndim ==2:
                    x = torch.unsqueeze(x,0)
                self.model.zero_grad()
                output,mu,logvar = self.model(x)
                MSE = self.loss(x[:,-1,:], output)
                KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
                l = MSE + self.theta * KLD
                l.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.losses["train"].append(l.detach().cpu().numpy())
            # self.schedular.step()
            train_loss = np.array(self.losses["train"]).mean()
            if self.val_loader is None:
                print("[epoch :%d/%d] train_loss:%f" % (
                epoch+1, self.n_epochs, train_loss))
            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for x, y in tqdm(self.val_loader):
                        x, y = x.to(self.device), y.to(self.device)
                        output,mu,logvar = self.model(x)
                        MSE = self.loss(x[:,-1,:],output)
                        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
                        l = MSE + self.theta * KLD
                        self.losses["val"].append(l.detach().cpu().numpy())
                val_loss = np.array(self.losses["val"]).mean()
                print("[epoch :%d/%d] train_loss:%f,val_loss:%f" % (epoch+1, self.n_epochs, train_loss,val_loss))
                if self.val_min_loss is None:
                    self.val_min_loss = val_loss
                if val_loss <= self.val_min_loss:
                    self.save_model(self.save_path+"/model.pth",self.model)
                    self.val_min_loss=val_loss
                self.losses["val"], self.losses["train"] = [], []
        if self.val_loader is None:
            self.save_model(self.save_path + "/model.pth", self.model)
        end_time = time.time()
        print("The train average time is:",(end_time-start_time)/self.n_epochs)

    def train_scores(self):
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                recon,mu,logvar = self.model(x)
                # MSE = torch.sum(torch.sqrt((x[:,-1,:] - recon) ** 2 + 1e-5), dim=1)
                # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                MSE = torch.sqrt((x[:, -1, :] - recon) ** 2 + 1e-5)
                KLD = -0.5 * torch.as_tensor(1 + logvar - mu.pow(2) - logvar.exp())
                scores = MSE + self.theta * KLD

                self.losses["train_scores"].append(scores.detach().cpu().numpy())
    def test_scores(self):
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                recon,mu,logvar = self.model(x)
                # MSE = torch.sum(torch.sqrt((x[:,-1,:] - recon) ** 2 + 1e-8), dim=1)
                # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                MSE = torch.sqrt((x[:, -1, :] - recon) ** 2 + 1e-5)
                KLD = -0.5 * torch.as_tensor(1 + logvar - mu.pow(2) - logvar.exp())
                scores = MSE + self.theta * KLD
                self.losses["test_scores"].append(scores.detach().cpu().numpy())

    def anomaly_detection(self, labels):
        train_scores = np.concatenate(self.losses['train_scores'])
        test_scores = np.concatenate(self.losses['test_scores'])
        if self.dataset=="SYNTHETIC":
            for i in range(test_scores.shape[1]):
                train_score,test_score,label,value = train_scores[:,i],test_scores[:,i],labels[:,i],self.testdata[:,i]
                pot_result, _ = pot_eval(train_score, test_score, label, self.dataset)
                self.plot_tn(test_score,label,value,pot_result["threshold"],i)
            train_scores = np.sum(train_scores,axis=1)
            test_scores = np.sum(test_scores,axis=1)
            labels = np.array(np.sum(labels,axis=1),dtype=np.bool).astype(int)
        pot_result,pred_labels = pot_eval(train_scores, test_scores, labels, self.dataset)
        pot_result.update(hit_att(test_scores, labels))
        pot_result.update(ndcg(test_scores, labels))

        # The DNT threshlod
        DNT_eval = epsilon_eval(train_scores, test_scores, labels, reg_level=reg_level[self.dataset])

        # The best F1-scores method
        bf_eval = bf_search(test_scores, labels, start=0, end=15, step_num=200, verbose=False)
        self.plot_(test_scores,labels,pot_result["threshold"])
        bf_eval_without = bf_without_point_adjust_search(test_scores, labels, start=0, end=15, step_num=200, verbose=False)
        print(f"Results using peak-over-threshold method:\n {pot_result}")
        print(f"Results using DNT method:\n {DNT_eval}")
        print(f"Results using best f1 score search:\n {bf_eval}")
        print(f"Results using best f1 score search without point adjus:\n {bf_eval_without}")
        summary = f"epsilon_result \n {DNT_eval}\n, pot_result: \n{pot_result}\n," \
                  f""f" bf_result: \n{bf_eval}\n,bf_result_without_point_adjust: \n{bf_eval_without}\n,"
        with open(f"{self.save_path}/{self.summary_file_name}", 'w') as f:
            f.write(str(summary))
            f.close()
    def save_model(self,save_path, model):
        if torch.cuda.device_count() == 1:
            torch.save(model.state_dict(), save_path)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), save_path)
    def load_model(self,save_path, model,device,model_name="/model.pth"):
        dir_content = os.listdir(save_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{save_path}/{subf}") and subf != "logs"]
        date_times = [datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')
        self.save_path = save_path + "/" + model_id
        PATH = save_path + "/" + model_id + model_name
        state_dict = torch.load(PATH)
        if torch.cuda.device_count() == 1:
            model.load_state_dict(state_dict)
        if torch.cuda.device_count() > 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        model.to(device)
    def plot_(self,test_scores,labels,threshold):
        pass
        # plt.subplot(211)
        # plt.axhline(y=threshold,color="red",ls=":")
        # plt.plot(test_scores,label="anomalyScore")
        # plt.xlim((0,5200))
        #
        # plt.subplot(312)
        # plt.plot(self.testdata[:,],label="feature0")
        # plt.xlim((0, 5200))
        #
        # plt.subplot(212)
        # plt.plot(labels, label="lable")
        # plt.xlim((0, 5200))
        # plt.legend()
        # plt.show()
        # plt.savefig(self.save_path + "/AnomalyScores.svg")
    def plot_tn(self,test_scores,label,value,threshold,i):
        plt.rc('font', family='Times New Roman', size=9)
        plt.figure(figsize=(12, 1.2))
        plt.ylim((0, 1.2))
        plt.plot(test_scores, zorder=1,label="Anomaly Scores")
        plt.axhline(y=threshold,color="r", zorder=2,label="Threshold")
        plt.fill_between(np.arange(label.shape[0]), label.astype(np.int8), color="yellow",alpha=0.4,
                         zorder=3,label="True Anomaly")


        plt.title(f'Dimension = {i}',x=0.1,y=0.7)
        plt.xlabel('Timestamp', fontsize=9)
        plt.ylabel('Value', fontsize=9)
        plt.legend(ncol=3,bbox_to_anchor=(0.3,1.12))

        plt.savefig("./image/syn/"+str(i)+"dimension.svg",bbox_inches='tight')





