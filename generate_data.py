import torch
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class generate_data():
    """
    class for generate the EIS data based on the Universal circuit.
    Data is generated randmoly within defined distributions as described in the paper. 
    """
    def __init__(self, size=3e6, w_lb=-3, w_ub=8, samples=200):
        np.random.seed(0)
        self.size = size
        self.w = np.logspace(w_lb, w_ub, samples)
        self.ind_lb = 1e-6
        self.ind_ub = 2e-4
        self.warburg_lb = 80
        self.warburg_ub = 160
        self.R_lb = 10
        self.R_ub = 1e6
        self.C_lb = 1e-7
        self.C_ub = 1e-4
        self.P_list = [0.2, 0.2, 0, 0.5, 0.5, 0.5]  # probability of the binary elements

    def forward_model1(self, w, params, inductance,
                       Warburg_p, Warburg_s, R_l, L,
                       R_sub, C_sub, binaries, C_series):
        w = w*2*np.pi
        
        def Z_w(Warburg, w):
            Z_w = (Warburg/(w**0.5)) * (1-1j)
            return Z_w
        
        def Z_RC(w, R_p, C_p):
            return R_p/(1 + (w*R_p*C_p*1j))

        def Z_L(w, L):
            return w*L*1j
        
        def Z_C(w, C):
            return 1/(C*w*1j)
        
        def Z_R(R):
            return R
        
        def Z_RC_w_RL_RC(w, R_p, C_p,
                         W_p, R_l, L,
                         R_sub, C_sub, binaries):
            return 1/( \
                1/Z_C(w,C_p)\
                + 1/(Z_R(R_p) + (Z_w(W_p,w)*binaries[:,3])+ (binaries[:,4]*Z_RC(w,R_sub,C_sub)))\
                    + ( (1/(Z_R(R_l) + Z_L(w,L))) *binaries[:,5])\
                                )
        Z = params[0,0,:] +Z_C(w[:,None],C_series)*binaries[:,0] + Z_L(w[:,None],inductance)*binaries[:,1]+ Z_w(Warburg_s,w[:,None])*binaries[:,2] + \
            Z_RC_w_RL_RC(w[:,None],params[0,1,:],params[1,1,:],Warburg_p,R_l,L,R_sub,C_sub,binaries) \
                +Z_RC(w[:,None],params[0,2,:],params[1,2,:]) \
                    +Z_RC(w[:,None],params[0,3,:],params[1,3,:])
        
        Z = Z.T
        Z_real = np.real(Z)
        Z_imag = -np.imag(Z)

        return Z_real, Z_imag

    def zeros_poly1(self, C, degree):
        hot_vec = np.zeros((C.shape[2], C.shape[1]))

        degs = []
        for i in range(C.shape[2]):
            deg = np.random.randint(2, degree+1)
            degs.append(deg)
            C[:,deg:,i] = C[:,deg:,i]*0
            hot_vec[i,deg-1] = 1
        return C, np.array(degs), hot_vec


    def generate(self):
        w = self.w
        D_size = self.size
        degree = 4 
        tau_ratio_before = 1
        tau_ratio = tau_ratio_before
        tau_ratio = np.sqrt(tau_ratio)
        scaling_vec = tau_ratio ** np.arange(degree-1)
        scaling_vec = np.append(1, scaling_vec)
        R_lb_before_scaling = self.R_lb
        R_ub_before_scaling = self.R_ub
        C_lb_before_scaling = self.C_lb
        C_ub_before_scaling = self.C_ub
        R_lb = np.ones(degree)*R_lb_before_scaling
        R_ub = np.ones(degree)*R_ub_before_scaling
        R_ub = R_ub*scaling_vec
        R_lb = R_lb*scaling_vec
        R_lb = np.tile(R_lb.reshape(-1,1), D_size)
        R_ub = np.tile(R_ub.reshape(-1,1), D_size)
        C_lb = np.ones(degree)*C_lb_before_scaling
        C_ub = np.ones(degree)*C_ub_before_scaling
        C_ub = C_ub*scaling_vec
        C_lb = C_lb*scaling_vec
        C_lb = np.tile(C_lb.reshape(-1,1), D_size)
        C_ub = np.tile(C_ub.reshape(-1,1), D_size)
        ind_lb = self.ind_lb
        ind_ub = self.ind_ub
        warburg_lb = self.warburg_lb
        warburg_ub = self.warburg_ub

        Resisitors = np.random.uniform(R_lb, R_ub, (degree,D_size))
        Capacitors = np.random.uniform(C_lb, C_ub, (degree,D_size))
        C_series = np.random.uniform(self.C_lb, self.C_ub, (D_size))
        Inductance = np.random.uniform(ind_lb, ind_ub, (D_size))
        Inductance_p = np.random.uniform(ind_lb, ind_ub, (D_size))
        R_Inductance_p = np.random.uniform(R_lb_before_scaling,R_ub_before_scaling, (D_size))
        R_SubRC = np.random.uniform(self.R_lb, self.R_ub, (D_size))
        C_SubRC = np.random.uniform(C_lb_before_scaling, C_ub_before_scaling, (D_size))
        Warburg_parallel = np.random.uniform(warburg_lb, warburg_ub, (D_size))
        Warburg_series = np.random.uniform(warburg_lb, warburg_ub, (D_size))

        params = np.array([Resisitors, Capacitors])
        binary_names=  ['Cs', 'Ls', 'Ws', 'wp', 'RCp', 'RLp']
        B_N = len(binary_names)
        P_list = self.P_list
        Cs_binary = np.random.choice([0, 1],
                                     size=(D_size),
                                     p=[1-P_list[0],
                                     P_list[0]]).reshape(-1, 1)
        
        Ls_binary = np.random.choice([0, 1],
                                     size=(D_size),
                                     p=[1-P_list[1],
                                     P_list[1]]).reshape(-1, 1)
        Ws_binary = np.random.choice([0, 1],
                                     size=(D_size),
                                     p=[1-P_list[2],
                                     P_list[2]]).reshape(-1, 1)
        Wp_binary = np.random.choice([0, 1],
                                     size=(D_size),
                                     p=[1-P_list[3],
                                     P_list[3]]).reshape(-1, 1)
        RCp_binary = np.random.choice([0, 1],
                                      size=(D_size),
                                      p=[1-P_list[4],
                                      P_list[4]]).reshape(-1, 1)
        RLp_binary = np.random.choice([0, 1],
                                      size=(D_size),
                                      p=[1-P_list[5],
                                      P_list[5]]).reshape(-1, 1)
        Binary_Matrix = np.hstack((Cs_binary, Ls_binary,
                                   Ws_binary, Wp_binary,
                                   RCp_binary, RLp_binary))
        params_f = np.copy(params)
        params_zeros, degs, hot_vector_orig = self.zeros_poly1(params_f, degree) # RC Degree, indicates how many RCs in the circuit
        onehot_vector = hot_vector_orig[:, 1:]
        Z_real, Z_imag = self.forward_model1(w, params_zeros,
                                             Inductance, Warburg_parallel,
                                             Warburg_series, R_Inductance_p,
                                             Inductance_p, R_SubRC, C_SubRC,
                                             Binary_Matrix, C_series)
        w_wide = np.logspace(-6, 8, 200)
        Z_real_wide, Z_imag_wide = self.forward_model1(w_wide, params_zeros,
                                                      Inductance, Warburg_parallel,
                                                      Warburg_series, R_Inductance_p,
                                                      Inductance_p, R_SubRC, C_SubRC,
                                                      Binary_Matrix, C_series)
        Z_real_og_wide = np.copy(Z_real_wide)
        Z_imag_og_wide = np.copy(Z_imag_wide)

        return (w, Z_real,
                Z_imag, Z_real_og_wide,
                Z_imag_og_wide, onehot_vector,
                Binary_Matrix, degs)


