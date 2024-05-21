
import numpy as np
from impedance.models.circuits import CustomCircuit
import impedance
from impedance import preprocessing

import warnings
warnings.filterwarnings("ignore")

class MultiStart():
    """
    A module for defining the bounded multistart optimization method. This module uses the package impedance.
    The optimization method used is the trust region method.
    
    """
    def __init__(self):
        np.random.seed(0)
        self.ind_lb = 1e-6
        self.ind_ub = 2e-4
        self.warburg_lb = 80
        self.warburg_ub = 160
        self.R_lb = 10
        self.R_ub = 1e6
        self.C_lb = 1e-7
        self.C_ub = 1e-4



    def MAPE(self,Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape

    def write_cricuit(self,deg,binary):
        circ = 'R0' + (binary[1] * '-C0') + (binary[2] * '-L0')
        circ = circ+ '-p(R1' + binary[0]*'-W0' + ',C1)'
        for i in range(2,deg):
            R = 'R'+ str(i)
            C = 'C'+ str(i)
            circ = circ+ '-p('+R+','+C+')'
        return circ

    def get_init_guess(self,deg,binary):
        lb = [self.R_lb]
        ub = [self.R_ub]
        if bool(int(binary[1])):
            lb.append(self.C_lb)
            ub.append(self.C_ub)
        if bool(int(binary[2])):
            lb.append(self.ind_lb)
            ub.append(self.ind_ub)
        lb.append(self.R_lb)
        ub.append(self.R_ub)
        if bool(int(binary[0])):
            lb.append(self.warburg_lb)
            ub.append(self.warburg_ub)
        lb.append(self.C_lb)
        ub.append(self.C_ub)
        for i in range(2,deg):
            lb.append(self.R_lb)
            lb.append(self.C_lb)
            ub.append(self.R_ub)
            ub.append(self.C_ub)
        init_guess = np.random.uniform(lb,ub)
        return init_guess


    def get_bounds(self,deg,binary):
        lb = [self.R_lb]
        ub = [self.R_ub]
        if bool(int(binary[1])):
            lb.append(self.C_lb)
            ub.append(self.C_ub)
        if bool(int(binary[2])):
            lb.append(self.ind_lb)
            ub.append(self.ind_ub)
        lb.append(self.R_lb)
        ub.append(self.R_ub)
        if bool(int(binary[0])):
            lb.append(self.warburg_lb)
            ub.append(self.warburg_ub)
        lb.append(self.C_lb)
        ub.append(self.C_ub)
        for i in range(2,deg):
            lb.append(self.R_lb)
            lb.append(self.C_lb)
            ub.append(self.R_ub)
            ub.append(self.C_ub)
        return lb,ub

    #***************************** Optimzer ****************************

    def multistart_min(self,w,Z_real, Z_imag, N,degree,binary,Bounds=False):
        parameters_list = []
        error_per_list = [] 
        degree = int(degree)
        for i in range(N):
            try:
                if Bounds==True:
                    bnds = (tuple(self.get_bounds(degree,binary)[0]), tuple(self.get_bounds(degree,binary)[1]))   
                else:
                    bnds=None
                frequencies, Z = w.reshape(-1,1), (Z_real - (Z_imag *1j)).reshape(-1,1)
                circuit = self.write_cricuit(degree,binary)
                initial_guess = self.get_init_guess(degree,binary)
                Z_mes = Z
                circuit = CustomCircuit(circuit, initial_guess=initial_guess)
                frequencies, Z = frequencies.flatten(), Z.flatten()

                circuit.fit(frequencies, Z,maxfev=1000,bounds=bnds)
                Z_fit = circuit.predict(frequencies)
                error_per_list.append(self.MAPE(Z_mes.reshape(-1,1),Z_fit.reshape(-1,1)))
                parameters_list.append(parameters_list)
            except:
                error_per_list.append(np.nan)
                parameters_list.append(0)
        
        return np.array(error_per_list),parameters_list


