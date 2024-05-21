import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import joblib
from generate_data import generate_data

warnings.filterwarnings("ignore")


class pre_process():
    '''
    Prepare the spectra (data) for training the neural network:
    1- standardization of the data.
    2- Reshaping the data as input for the neural network
    3- Splitting the data for training, validation, and testing (0.7,0.2,0.1)
    4- Splitting test data for curve fitting
    '''
    def __init__(self, scale=True, scale_type='stand', device=0,
                  New_Training=False,
                  data=None,
                  Neual_Network='net1'):
        '''
        args:
        scale (Bool): Data is scaled if the value is True   
        scale_type (str): scaling type; normalization or min-max scaling
        device (int): cuda decive number if available
        data (numpy): the traing data
        Neural_Network 'str': the choice of the neural network, net1 or net2 (see Module NN.py)  
        '''
        self.w = data[0]
        self.Z_real = data[1]
        self.Z_imag = data[2]
        self.Z_real_wide = data[3]
        self.Z_imag_wide = data[4]
        self.onehot_vector = data[5]
        self.Binary_Matrix = data[6]
        self.degs = data[7]
        self.scale = scale
        self.scale_type = scale_type
        self.model = Neual_Network

        self.New_Training = New_Training
        self.generator = generate_data()
        self.ind_ub = self.generator.ind_ub

    def process(self):
        '''
        the pre-processing function.
        returns:
            1- Tuple containing the training the validation set
            2- Tuple of test data
            3- Tuple of data for the curve fitting part
        '''
        Norm_type = self.scale_type
        hot_vector = self.onehot_vector
        Binary_Matrix = self.Binary_Matrix
        Z_real = self.Z_real
        Z_imag = self.Z_imag
        degs = self.degs
        Z_real_og = np.copy(Z_real)
        Z_imag_og = np.copy(Z_imag)
        ind_ub = self.ind_ub
        Z_imag = (Z_imag + ind_ub*2*np.pi*self.w.max())

        Z_real, Z_imag = np.log(Z_real), np.log(Z_imag)
        if self.New_Training:
            if Norm_type == 'minmax':
                scaler_real = MinMaxScaler()
                scaler_imag = MinMaxScaler()
            elif Norm_type == 'stand':
                scaler_real = StandardScaler()
                scaler_imag = StandardScaler()
            scaler_real.fit(Z_real)
            scaler_imag.fit(Z_imag)
            Z_real = scaler_real.transform(Z_real)
            Z_imag = scaler_imag.transform(Z_imag)
            joblib.dump(scaler_real, 'scaler_real.gz')
            joblib.dump(scaler_imag, 'scaler_imag.gz')
        else:
            scaler_real = joblib.load('scaler_real.gz')
            scaler_imag = joblib.load('scaler_imag.gz')
            Z_real = scaler_real.transform(Z_real)
            Z_imag = scaler_imag.transform(Z_imag)
        
        input_ = np.array((Z_real, Z_imag, Z_real_og, Z_imag_og))
        input_ = np.swapaxes(input_, 0, 1)
        if self.model == 'net1':
            input_ = input_.reshape(input_.shape[0], 1,
                                     input_.shape[1], input_.shape[2])
        
        split = 0.8
        train_data = input_[:int(input_.shape[0]*split)].astype(float)
        test_data = input_[int(input_.shape[0]*split):].astype(float)
        hot_vec_train = hot_vector[:int(input_.shape[0]*split)]
        hot_vec_test = hot_vector[int(input_.shape[0]*split):]
        binary_train = Binary_Matrix[:int(input_.shape[0]*split)]
        binary_test = Binary_Matrix[int(input_.shape[0]*split):]
        Z_real_test = Z_real_og[int(input_.shape[0]*split):]
        Z_imag_test = Z_imag_og[int(input_.shape[0]*split):]
        Z_real_test_wide = self.Z_real_wide[int(input_.shape[0]*split):]
        Z_imag_test_wide = self.Z_imag_wide[int(input_.shape[0]*split):]
        degs_test = degs[int(input_.shape[0]*split):].astype(float)




        train_set = torch.utils.data.TensorDataset(
                    torch.from_numpy(train_data),
                    torch.from_numpy(hot_vec_train),
                    torch.from_numpy(binary_train))

               
        test_set = torch.utils.data.TensorDataset(
                    torch.from_numpy(test_data),
                    torch.from_numpy(hot_vec_test),
                    torch.from_numpy(binary_test))


        split_train = 0.7
        split_val = 0.2
        split_test = 0.1

        num_samples = input_.shape[0]
        train_end = int(num_samples * split_train)
        val_end = train_end + int(num_samples * split_val)

        train_data = input_[:train_end].astype(float)
        val_data = input_[train_end:val_end].astype(float)
        test_data = input_[val_end:].astype(float)

        hot_vec_train = hot_vector[:train_end]
        hot_vec_val = hot_vector[train_end:val_end]
        hot_vec_test = hot_vector[val_end:]

        binary_train = Binary_Matrix[:train_end]
        binary_val = Binary_Matrix[train_end:val_end]
        binary_test = Binary_Matrix[val_end:]

        Z_real_test = Z_real_og[val_end:]
        Z_imag_test = Z_imag_og[val_end:]
        Z_real_test_wide = self.Z_real_wide[val_end:]
        Z_imag_test_wide = self.Z_imag_wide[val_end:]
        degs_test = degs[val_end:].astype(float)

        # Create datasets
        train_set = torch.utils.data.TensorDataset(
            torch.from_numpy(train_data),
            torch.from_numpy(hot_vec_train),
            torch.from_numpy(binary_train)
        )

        val_set = torch.utils.data.TensorDataset(
            torch.from_numpy(val_data),
            torch.from_numpy(hot_vec_val),
            torch.from_numpy(binary_val)
        )

        test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(test_data),
            torch.from_numpy(hot_vec_test),
            torch.from_numpy(binary_test)
        )


        return  (train_set, val_set), \
                (test_data, Z_real_test, Z_imag_test, degs_test, binary_test), \
                (Z_real_test_wide, Z_imag_test_wide)    