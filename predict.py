import torch
import numpy as np
import warnings
from NN import NN1 as Net1
from NN import NN2 as Net2
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")


class Prediction():
    """
    A module for testing the neural network:
    1- predicting the the circuits from the EIS data
    2- plotting results of the test; ex. confusion matrix
    3- determing the class probabilities and plotting the top-k accuracy
    """
    def __init__(self,
                 Test_Data=None,
                 device='cuda:0',
                 Neual_Network='net1'):
        
        torch.cuda.set_device(device)
        if Neual_Network == 'net1':
        
            self.net = Net1().to(device)
        if Neual_Network == 'net2':
            self.net = Net2().to(device)

        self.test_dataset = Test_Data
        self.device = device

    def predict(self):
        net = self.net

        test_data, Z_real_test, Z_imag_test, degs_test, binary_test = self.test_dataset
        tst_idx = min(2000,len(test_data))
        net.load_state_dict(torch.load('best-model-parameters_run9.pt',
                                       map_location=self.device))

        Confidence , degree_prediction, binary_prediction = net(torch.from_numpy(test_data[:tst_idx]).type(torch.FloatTensor).to(self.device))
        Confidence = Confidence.cpu().detach().numpy()
        degree_prediction = degree_prediction.cpu().detach().numpy()
        binary_prediction = binary_prediction.cpu().detach().numpy()

        np.savetxt('predictions/Pred_Binary_prob', binary_prediction)
        np.savetxt('predictions/Pred_Binary', np.round(binary_prediction))

        np.savetxt('predictions/Pred_Deg_prob', Confidence)
        np.savetxt('predictions/Pred_Deg', degree_prediction)

        np.savetxt('predictions/True_Binary', binary_test)
        np.savetxt('predictions/True_Deg', degs_test)
        np.savetxt('predictions/Test_Z_real', Z_real_test)
        np.savetxt('predictions/Test_Z_imag', Z_imag_test)
        
        binary_prob = np.copy(binary_prediction)
        binary_prediction = np.round(binary_prediction)
        binary_prediction_capacitor = binary_prediction[:,0]
        binary_prediction_inductance= binary_prediction[:,1]
        binary_prediction_warburg_parallel = binary_prediction[:,3]
        binary_prediction_SubRC = binary_prediction[:,4]
        binary_prediction_SubRL = binary_prediction[:,5]
        all_preD = np.hstack((degree_prediction.reshape(-1,1), binary_prediction))
        all_True = np.hstack((degs_test[:tst_idx].reshape(-1,1), binary_test[:tst_idx]))
        acc = (all_preD==all_True).all(axis=1).sum() / tst_idx

        Con_Mat = sklearn.metrics.confusion_matrix(degs_test[:tst_idx],degree_prediction)
        Con_Mat_binary1 = sklearn.metrics.confusion_matrix(binary_test[:tst_idx,0],binary_prediction_capacitor)
        Con_Mat_binary2 = sklearn.metrics.confusion_matrix(binary_test[:tst_idx,1],binary_prediction_inductance)
        Con_Mat_binary3 = sklearn.metrics.confusion_matrix(binary_test[:tst_idx,3],binary_prediction_warburg_parallel)
        Con_Mat_binary4 = sklearn.metrics.confusion_matrix(binary_test[:tst_idx,4],binary_prediction_SubRC)
        Con_Mat_binary5 = sklearn.metrics.confusion_matrix(binary_test[:tst_idx,5],binary_prediction_SubRL)

        fig, ax = plt.subplots(3,2,figsize=(12,12))
        fig.suptitle('Confusion Matrices',fontsize=18)
        ann = []
        for i in Con_Mat:
            for j in i:
                ann.append(str(j) + '\n' + str(np.round(j/sum(i),2)))
        ann=np.asarray(ann).reshape(3,3)

        Con_Mat = Con_Mat/Con_Mat.sum(axis=1)
        Con_Mat_df = pd.DataFrame(Con_Mat)
        Con_Mat_df.columns=['1 RC', '2 RCs', '3 RCs']
        Con_Mat_df.index=['1 RC', '2 RCs', '3 RCs']
        sns.heatmap(Con_Mat_df,
                    annot=ann,ax=ax[0,0],
                    cmap='Blues', fmt='',
                    xticklabels=True,yticklabels=True)
        
        ax[0,0].set_title('RC Degree')
        ann = []
        for i in Con_Mat_binary1:
            for j in i:
                ann.append(str(j) + '\n' + str(np.round(j/sum(i),2)))
        ann=np.asarray(ann).reshape(2,2)
        Con_Mat_binary1 = Con_Mat_binary1/Con_Mat_binary1.sum(axis=1)
        sns.heatmap(Con_Mat_binary1,annot=ann,
                    ax=ax[0,1],cmap='Blues', fmt='')
        
        ax[0,1].set_title(r'Series Capacitor $C_{s}$')
        ann = []
        for i in Con_Mat_binary2:
            for j in i:
                ann.append(str(j) + '\n' + str(np.round(j/sum(i),2)))
        ann=np.asarray(ann).reshape(2,2)

        Con_Mat_binary2 = Con_Mat_binary2/Con_Mat_binary2.sum(axis=1)
        sns.heatmap(Con_Mat_binary2,annot=ann, ax=ax[1,0],cmap='Blues', fmt='')
        ax[1,0].set_title(r'Series Inductance $L_{s}$')

        ann = []
        for i in Con_Mat_binary3:
            for j in i:
                ann.append(str(j) + '\n' + str(np.round(j/sum(i),2)))
        ann=np.asarray(ann).reshape(2,2)

        Con_Mat_binary3 = Con_Mat_binary3/Con_Mat_binary3.sum(axis=1)
        sns.heatmap(Con_Mat_binary3,annot=ann,
                    ax=ax[1,1], cmap='Blues', fmt='')
        
        ax[1,1].set_title(r'Warburg $W$')
        ann = []
        for i in Con_Mat_binary4:
            for j in i:
                ann.append(str(j) + '\n' + str(np.round(j/sum(i),2)))
        
        ann=np.asarray(ann).reshape(2,2)
        Con_Mat_binary4 = Con_Mat_binary4/Con_Mat_binary4.sum(axis=1)
        ax[2,0].set_title(r'$R_{p_1}$ - $C_{p}$')
        sns.heatmap(Con_Mat_binary4, annot=ann,
                    ax=ax[2,0], cmap='Blues', fmt='')
        
        ann = []
        for i in Con_Mat_binary5:
            for j in i:
                ann.append(str(j) + '\n' + str(np.round(j/sum(i),2)))
        ann=np.asarray(ann).reshape(2,2)

        Con_Mat_binary5 = Con_Mat_binary5/Con_Mat_binary5.sum(axis=1)
        sns.heatmap(Con_Mat_binary5,annot=ann, ax=ax[2,1],cmap='Blues', fmt='')
        ax[2,1].set_title(r'$R_{p_2}$ - $L_{p}$')
        fig.savefig('predictions/ALLCon.png', transparent=False,facecolor='white')
        plt.close()
        
        confidence_list = np.linspace(0.35, .95, 20)
        accVSconf = []
        for conf in confidence_list:
            idx = Confidence.max(1)>conf
            degs_true_conf = degs_test[:tst_idx][idx]
            degs_pred_conf = degree_prediction[idx]
            accVSconf.append(accuracy_score(degs_true_conf,degs_pred_conf))


        elem_binary=np.zeros((2**5, 5))
        circuits_binary=np.zeros((3 * 2**5, 6))
        for j in range(0,2**5):
            BIN = np.array(list('{0:05b}'.format(j)))
            elem_binary[j] = BIN
        deg_vec = np.tile(np.array([0,1,2]),int(96/3))
        elem_binary = np.tile(elem_binary,(3,1))
        circuits_binary[:,0] = deg_vec
        circuits_binary[:,1:] =elem_binary

        true_data = np.zeros((tst_idx,6))
        true_data[:,0] = degs_test[:tst_idx]-2
        true_data[:,1:] = binary_test[:tst_idx,[0,1,3,4,5]]
        y_true = []
        for i in range(tst_idx):
            y = int(np.where((true_data[i]==circuits_binary).all(axis=1)==True)[0])
            y_true.append(y)
        y_true=np.array(y_true)

        classes_prob = np.zeros((tst_idx,3 * 2**5))
        for j in range(tst_idx):
            prob_vec = []
            for i in range(circuits_binary.shape[0]):
                prob = circuits_binary[i,1:] *binary_prob[j,[0,1,3,4,5]]  +  ((1-circuits_binary[i,1:] )* (1-binary_prob[j,[0,1,3,4,5]]))#np.maximum(((1-circuits_binary[i,1:] )* binary_prob[j,[0,1,3,4,5]]),0)
                prob = np.prod(prob)
                prob = Confidence[j][int(circuits_binary[i,0])] * prob
                prob_vec.append(prob)

            prob_vec = np.array(prob_vec)
            classes_prob[j] = prob_vec


        from sklearn.metrics import top_k_accuracy_score
        y_score =np.copy(classes_prob)

        #print(y_score.shape)
        print(y_true)

        acc_list=[]
        k_list = np.arange(1,11)
        for i in k_list:
            acc_list.append(top_k_accuracy_score(y_true, y_score, k=i,labels=np.arange(0,96,1)))
        acc_list = np.array(acc_list)
        plt.rcParams['font.size'] = '16'

        plt.figure(figsize=(9,9))
        plt.plot(k_list,np.array(acc_list)*100,c='black')
        plt.xticks(k_list)
        plt.grid(visible=True, linestyle='-', linewidth=2)
        plt.title('Accuracy given Top-K')
        plt.ylabel('Accuracy %')
        plt.xlabel('Top-k')
        plt.savefig('predictions/AccuracyVstopK.png', transparent=False,facecolor='white')
        plt.close()

        top3 = classes_prob.T.argsort(axis=0)[-3:][::-1].T

        np.savetxt('predictions/Top3',top3)
        np.savetxt('predictions/CircuitsBinaryEncoding',circuits_binary)


