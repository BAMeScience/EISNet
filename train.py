import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms

import warnings
from NN import NN1 as Net1
from NN import NN2 as Net2
warnings.filterwarnings("ignore")


class train():
    """
    A module for training the neural network.
    """
    def __init__(self,Training_data,device=0 ,
                 continue_training=False,batch_size = 512,
                 learning_rate=1e-4,epochs=700,
                 Neual_Network = 'net1'):
        if Neual_Network == 'net1':        
            self.net = Net1().to(device)
        if Neual_Network == 'net2':
            self.net = Net2().to(device)

        self.train_set, self.test_set = Training_data 

        self.best_loss = 10000
        self.continue_training = continue_training
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.device = device



    def setup_data_loaders(self, batch_size=128, use_cuda=False):
        root = './data'
        download = True
        trans = transforms.ToTensor()

        kwargs = {'pin_memory': use_cuda}
        train_loader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                batch_size=batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_set,
                                                batch_size=batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader


    def start_training(self):
        print('Starting training')
        best_loss=self.best_loss
        net = self.net
        training_loss = []
        Valid_loss = []

        continue_training = self.continue_training
        if continue_training:
            net.load_state_dict(torch.load('best-model-parameters_run9.pt', map_location=self.device))
            losses = np.loadtxt('Training Loss/losses')
            training_loss = list(losses[0])
            Valid_loss = list(losses[1])

        if 'cuda' in self.device:
            use_cuda = True
        else:
            use_cuda = False

        train_loader,test_loader  = self.setup_data_loaders(
            batch_size=self.batch_size, use_cuda=use_cuda)

        opt_params = net.parameters()
        optimizer = torch.optim.Adam(opt_params, lr=self.lr)
        CEL =nn.CrossEntropyLoss(reduction='mean') # Loss metric for RC prediction
        BCE = nn.BCELoss(reduction='mean') # Loss metric for the binary elements; Warburg, inductance, etc.

        def my_loss(target_d,output_d,target_b,output_b):
            loss =  CEL(target_d,output_d)  + (BCE(output_b,target_b))
            return loss
        
        criterion = my_loss
        epochs=self.epochs
        step_count = 0

        for epoch in range(epochs):
            loss = 0
            validation_loss = 0
            for batch_features,hot_vec,binary_vec in train_loader:
                binary_vec = binary_vec.type(torch.FloatTensor).to(self.device)
                batch_features = batch_features.type(torch.FloatTensor).to(self.device)
                optimizer.zero_grad()
                encoder_outputs_1 ,encoder_outputs_2,binary_output = net(batch_features)
                train_loss= criterion(hot_vec.to(self.device), encoder_outputs_1,binary_vec.to(self.device),binary_output)
                train_loss.backward()
                optimizer.step()
                
                loss += train_loss.item()
                step_count = step_count+1
            loss = loss / len(train_loader)
            training_loss.append(loss)
            
            with torch.set_grad_enabled(False):
                for val_batch_features,val_hot_vec,val_binary_vec in test_loader:
                    val_binary_vec = val_binary_vec.type(torch.FloatTensor).to(self.device)
                    val_batch_features = val_batch_features.type(torch.FloatTensor).to(self.device)
                    val_encoder_outputs_1 , val_encoder_outputs_2, binary_output_val = net(val_batch_features)
                    val_loss= criterion(val_hot_vec.to(self.device), val_encoder_outputs_1,val_binary_vec,binary_output_val)
                    validation_loss += val_loss.item()

                validation_loss = validation_loss / len(test_loader)
                Valid_loss.append(validation_loss)
 
            if validation_loss<best_loss:
                torch.save(net.state_dict(), 'best-model-parameters_run9.pt') 
                best_loss = validation_loss
            print("epoch : {}/{}, loss = {:.6f} , val_loss ={:.6f} ".format(epoch + 1, epochs, loss,validation_loss))

            T_loss = np.array(training_loss)
            V_loss = np.array(Valid_loss)
            losses =np.array((T_loss,V_loss))
            np.savetxt('Training Loss/losses', losses)
            plt.ioff()  
            plt.figure(figsize=(9,9))
            plt.plot(np.array(training_loss))
            plt.plot(np.array(Valid_loss))
            plt.title('Training Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.savefig('Training Loss/Loss.pdf')
            plt.close()
            # plt.show()



