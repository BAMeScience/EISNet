
import torch
import torch.nn as nn


import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)


class NN1(nn.Module):
    """
    Neural network configuration.
    The neural network consists of 6 Conv. layers in two branches (6x2 in total) 
    one branch takes the scaled data as input and the other takes the non-scaled data as an input.
    The ouputs of each branch are concatenated at the end and connected to two output layers:
        1- layer for predicting the RCs
        2- layer for predicting the binary elements in the circuit 
        (refer to paper)
    """

    def __init__(self,deg_out=4,bin_out=6):
        super().__init__()
        self.main_activation = nn.Tanh()
        self.main_activation2 = nn.Tanh()
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.tanh = nn.Tanh()
        self.sigmoid  = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(.15)
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=300,            
                kernel_size=(1, 7),              
                stride=1,                   
                padding=0,                  
            ),                              
            self.main_activation2,                       
            nn.AvgPool2d(kernel_size=(1, 2)),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(300, 250, (1,2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool2d((1,2)),                
        ) 
        self.conv3 = nn.Sequential(         
            nn.Conv2d(250, 200, (1, 2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool2d((1, 2)),                
        ) 
        self.conv4 = nn.Sequential(         
            nn.Conv2d(200, 100, (1, 3), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool2d((1, 3)),                
        ) 
        self.conv5 = nn.Sequential(         
            nn.Conv2d(100, 30, (1, 2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool2d((1, 2)),                
        ) 
        self.conv6 = nn.Sequential(         
            nn.Conv2d(30, 10, (1, 2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool2d((1, 2)),                
        ) 
        self.conv21 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=200,            
                kernel_size=(1, 3),              
                stride=1,                   
                padding=0,                  
            ),                              
            self.main_activation,                       
            nn.MaxPool2d(kernel_size=(1, 2)),    
        )
        self.conv22 = nn.Sequential(         
            nn.Conv2d(200, 140, (1, 2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool2d((1, 2)),                
        ) 

        self.conv23 = nn.Sequential(         
            nn.Conv2d(140, 100, (1, 2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool2d((1, 2)),                
        ) 
        self.conv24 = nn.Sequential(         
            nn.Conv2d(100, 70, (1, 3), 1, 0),     
            self.main_activation,                      
            nn.MaxPool2d((1, 3)),                
        ) 

        self.conv25 = nn.Sequential(         
            nn.Conv2d(70, 30, (1, 2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool2d((1, 2)),                
        ) 

        self.conv26 = nn.Sequential(         
            nn.Conv2d(30, 6, (1, 2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool2d((1, 2)),                
        ) 


        self.out = nn.Linear(32, deg_out-1) 
        self.pre_out_b = nn.Linear(8,8)
        self.out_b = nn.Linear(32,bin_out)

    def forward(self, x):
        self.test=torch.clone(x)
        x1 = x[:,:,:2,:]
        x_p = self.conv21(x1)
        x_p = self.conv22(x_p)
        x_p = self.conv23(x_p)
        x_p = self.conv24(x_p)
        x_p = self.conv25(x_p)
        x_p = self.conv26(x_p)
        x_p = x_p.view(x_p.size(0), -1) 

        x2 = x[:,:,2:,:]
        x = self.conv1(x2)
        x = self.conv2(x)       
        x = self.conv3(x)
        x = self.conv4(x)        
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)  
        x = torch.cat((x, x_p), 1)     
        self.x = torch.clone(x)
        self.output_b = self.sigmoid(self.out_b(x))
        output = self.out(x)
        self.degree = self.softmax(output)
        self.deg_decimal = torch.argmax(self.degree,axis=1)+2
        return self.degree,self.deg_decimal,self.output_b
    

class NN2(nn.Module):
    def __init__(self,deg_out=4,bin_out=6):
        super().__init__()
        self.main_activation = nn.Tanh()
        self.main_activation2 = nn.ReLU()
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.tanh = nn.Tanh()
        self.sigmoid  = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(.2)
        self.conv1 = nn.Sequential(         
            nn.Conv1d(
                in_channels=2,              
                out_channels=300,            
                kernel_size=(7),              
                stride=1,                   
                padding=0,                  
            ),                              
            self.main_activation2,                    
            nn.AvgPool1d(kernel_size=(2)),    
        )

        self.conv2 = nn.Sequential(         
            nn.Conv1d(300, 250, (2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv3 = nn.Sequential(         
            nn.Conv1d(250, 200, (2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv4 = nn.Sequential(         
            nn.Conv1d(200, 100, (3), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv5 = nn.Sequential(         
            nn.Conv1d(100, 30, (2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv6 = nn.Sequential(         
            nn.Conv1d(30, 15, (2), 1, 0),     
            self.main_activation2,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv21 = nn.Sequential(         
            nn.Conv1d(
                in_channels=2,              
                out_channels=200,            
                kernel_size=(3),              
                stride=1,                   
                padding=0,                  
            ),                              
            self.main_activation,                       
            nn.MaxPool1d(kernel_size=(2)),    
        )
        self.conv22 = nn.Sequential(         
            nn.Conv1d(200, 140, (2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv23 = nn.Sequential(         
            nn.Conv1d(140, 100, (2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv24 = nn.Sequential(         
            nn.Conv1d(100, 70, (3), 1, 0),     
            self.main_activation,                      
            nn.MaxPool1d((3)),                
        ) 
        self.conv25 = nn.Sequential(         
            nn.Conv1d(70, 30, (2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool1d((2)),                
        ) 
        self.conv26 = nn.Sequential(         
            nn.Conv1d(30, 10, (2), 1, 0),     
            self.main_activation,                      
            nn.MaxPool1d((2)),                
        ) 
        self.out = nn.Linear(25, deg_out-1) 
        self.pre_out_b = nn.Linear(8,8)
        self.out_b = nn.Linear(25,bin_out)
    
    def forward(self, x):
        self.test=torch.clone(x)
        x1 = x[:,:2,:]
        x_p = self.conv21(x1)
        x_p = self.conv22(x_p)
        x_p = self.conv23(x_p)
        x_p = self.conv24(x_p)
        x_p = self.conv25(x_p)
        x_p = self.conv26(x_p)
        x_p = x_p.view(x_p.size(0), -1) 
        x2 = x[:,2:,:]
        x = self.conv1(x2)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)       
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)        
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, x_p), 1)     
        self.x = torch.clone(x)
        self.output_b = self.sigmoid(self.out_b(x))
        output = self.out(x)
        self.degree = self.softmax(output)
        self.deg_decimal = torch.argmax(self.degree,axis=1)+2
        return self.degree, self.deg_decimal, self.output_b
