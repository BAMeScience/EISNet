import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import warnings
from Regressor import MultiStart
import time
warnings.filterwarnings("ignore")

class Evaluation():
    '''
    Class for evaluation of the multistart curve fitting method for fitting the 
    parameters of predicted circuit.
    '''
    def __init__(self, freq_lb=-6, freq_ub=8):
        '''
        Args:
            freq_lb (float): the logarithmic lower bound of the (later defind) frequency vector
            freq_ub (float): the logarithmic upper bound of the (later defind) frequency vector
        '''
        self.Z_real_true = np.loadtxt('predictions/Test_Z_real')
        self.Z_imag_true = np.loadtxt('predictions/Test_Z_imag')
        self.Binary_Matrix_true = np.loadtxt('predictions/True_Binary')
        self.degs_true = np.loadtxt('predictions/True_Deg') # RC Degree, indicates how many RCs in the circuit
        self.w = np.logspace(freq_lb, freq_ub, 200) # creating the frequency vector
        self.OPT = MultiStart() # calling the optimizer 

    def runPointsTest(self, test_set=100, N=300):
        """
        Fucntion for running the points test for determing the optimal number of random points needed for the multi-start optimization method.
        
        Args:
            test_set (int): size of the test set (how many different circuits).  
            N (int): the maximum number of random points for intializing the optimizer.

        Output:
        1- saves the error data for each circuit for each number of points
        2- plot the fitting error change with increasing the points
        """
        error_array = np.zeros((test_set, N))
        for i in range(test_set):
            error_array[i], _ = self.OPT.multistart_min(self.w,
                                self.Z_real_true[i],
                                self.Z_imag_true[i],
                                N,
                                self.degs_true[i],
                                self.Binary_Matrix_true[i])
            np.savetxt('Evaluation results/Starting_points_error', error_array)
        
        err_90 = np.loadtxt('Evaluation results/Starting_points_error')
        err_90 = err_90[:45]
        N_times = np.arange(10, 300, 10)
        err_list = []
        for i in N_times:
            err_list.append(np.nanmin(err_90[:, :i], 1))

        df_boxes = pd.DataFrame(err_list).T
        df_boxes.columns = N_times
        plt.figure(figsize=(9, 9))
        sns.boxplot(data=df_boxes, showfliers=False)
        plt.xlabel('Number of Starting Points')
        plt.ylabel('Relative Error')
        plt.title('Multistart Evaluation')
        plt.yscale('log')
        plt.save()
        plt.savefig('Evaluation results/Starting_points_error.png',
                    transparent=False, facecolor='white')
        
    def final_evaluation(self,
                         Z_real_test_wide=None,
                         Z_imag_test_wide=None,
                         N_evaluation=30, test_length=300):
        
        
        """
        Runs fitting tests on the predicted circuit and the true circuit.

        Args:
            Z_real_test_wide, Z_real_test_wide (numpy array): the real and imaginary spectra for testing,
            N_evaluation (int): the number of the intialization points for the optimizer for each circuit fitting 
            test_length (int): the number of test circuits used for the evaluation.

        Output:
            1- saves fitting error of the circuits (true and predicted)
        
        """
        degree_prediction = np.loadtxt('predictions/Pred_Deg')
        binary_prediction = np.loadtxt('predictions/Pred_Binary')
        top3 = np.loadtxt('predictions/Top3')
        circuits_binary = np.loadtxt('predictions/CircuitsBinaryEncoding')
        predication_error_evaluation = []  # appends error for predicted degree
        predication_error_evaluation_2 = []
        predication_error_evaluation_3 = []
        True_error_evaluation = []  # appends error for true degree
        deg_4_error = []       

        Z_real_test = np.array(Z_real_test_wide)
        Z_imag_test = np.array(Z_imag_test_wide)
        w = self.w
        # topk_error=np.zeros((test_length,3))
        time_array = np.zeros((test_length, 5))
        true_circuit = []
        pred_circuit = []
        for i in range(test_length):
            print(i)
            pred_deg_i = degree_prediction[i].astype('int')
            pred_binary_i = binary_prediction[i].astype('int')
            pred1_time = time.time()
            pred_error_array, pred_parameters_array, Z_mes, Z_fit_pred = self.OPT.multistart_min(w,
                                                                                                  Z_real_test[i],
                                                                                                  Z_imag_test[i],
                                                                                                  N_evaluation,
                                                                                                  pred_deg_i,
                                                                                                  pred_binary_i,
                                                                                                  Bounds=True)
            pred1_time = time.time() - pred1_time

            predication_error_evaluation.append(np.nanmin(pred_error_array))         
            pred_deg_i = (circuits_binary[top3[i, 1], 0] +2).astype('int')
            pred_binary_i = (circuits_binary[top3[i, 1], 1:]).astype('int')
            pred_binary_i = np.insert(pred_binary_i, 2, 0)
            pred2_time = time.time()
            pred_error_array, pred_parameters_array, Z_mes, Z_fit_pred =  self.OPT.multistart_min(w,
                                                                                                  Z_real_test[i],
                                                                                                  Z_imag_test[i],
                                                                                                  N_evaluation,
                                                                                                  pred_deg_i,
                                                                                                  pred_binary_i,
                                                                                                  Bounds=True)
            pred2_time =time.time() - pred2_time
            predication_error_evaluation_2.append(np.nanmin(pred_error_array))
            pred_deg_i = (circuits_binary[top3[i, 2], 0] +2).astype('int')
            pred_binary_i = (circuits_binary[top3[i, 2], 1:]).astype('int')
            pred_binary_i = np.insert(pred_binary_i, 2, 0)
            pred3_time = time.time()
            pred_error_array, pred_parameters_array, Z_mes, Z_fit_pred =  self.OPT.multistart_min(w,
                                                                                                  Z_real_test[i],
                                                                                                  Z_imag_test[i],
                                                                                                  N_evaluation,
                                                                                                  pred_deg_i,
                                                                                                  pred_binary_i,
                                                                                                  Bounds=True)
            pred3_time =time.time() -  pred3_time

            predication_error_evaluation_3.append(np.nanmin(pred_error_array))
            true_deg_i = self.degs_true[i].astype('int')
            true_binary_i = self.Binary_Matrix_true[i].astype('int')
            true_time = time.time()
            true_error_array, true_parameters_array,Z_mes,Z_fit_true =  self.OPT.multistart_min(w,
                                                                                                Z_real_test[i],
                                                                                                Z_imag_test[i],
                                                                                                N_evaluation,
                                                                                                true_deg_i,
                                                                                                true_binary_i,
                                                                                                Bounds=True)
            True_error_evaluation.append(np.nanmin(true_error_array))
            true_time = time.time() - true_time

            deg_4 = 4
            binary_111 = np.array([1, 1, 0, 1, 1, 1])
            UC_time = time.time()
            error_array, true_parameters_array, Z_mes, Z_fit_U = self.OPT.multistart_min(w,
                                                                                           Z_real_test[i],
                                                                                           Z_imag_test[i],
                                                                                           N_evaluation,
                                                                                           deg_4,
                                                                                           binary_111,
                                                                                           Bounds=False)
            deg_4_error.append(np.nanmin(error_array))
            UC_time = time.time() - UC_time

            true_circuit.append(np.append(true_deg_i, true_binary_i))
            pred_circuit.append(np.append(pred_deg_i, pred_binary_i))
            time_vec = np.array([pred1_time, pred2_time, pred3_time, true_time, UC_time])
            time_array[i] = time_vec
            np.savetxt('Evaluation results/Time array', time_array)
            np.savetxt('Evaluation results/True circuit', true_circuit)
            np.savetxt('Evaluation results/Predicted Ciruit', pred_circuit)
            error_results = np.array((predication_error_evaluation,
                                     predication_error_evaluation_2,
                                     predication_error_evaluation_3,
                                     True_error_evaluation, deg_4_error))
            np.savetxt('Evaluation results/error_results', error_results.T)

    def plot_results():
        """
        plotting the error values of both the true and the predicted circuit against each other.
        """
        Error_Results = np.loadtxt('Evaluation results/error_results').T
        Time_results = np.loadtxt('Evaluation results/Time array').T
        # pred_circuit = np.loadtxt('Evaluation results/Predicted Ciruit').T
        # true_circuit = np.loadtxt('Evaluation results/True circuit').T
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 13))
        points=[]
        for i in range(len(Error_Results[0])):
            if Error_Results[0][i]<=Error_Results[4][i]*1.2:
                col = 'cornflowerblue'
                label = 'Prediciotn is better'
            else: 
                col='maroon'
                # label='Prediciotn is worse'

            point = ax1.scatter(Error_Results[0][i], Error_Results[4][i], c=col)
            points.append(point)
        ax1.set_title('Fitting Error of Predicted Circuit vs. Universal Circuit Fitting ')
        ax1.grid(visible=True)
        ax1.legend([points[0],points[9]],['Prediciotn is better', 'Prediciotn is worse'])
        ax1.set_xlabel('Fitting Error of Predicted Circuit [%]')
        ax1.set_ylabel('Fitting Error of Universal Circuit without reduction[%]')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # plt.savefig('PredvsU.png', transparent=False,facecolor='white')
        # plt.show()
        for i in range(len(Error_Results[0])):
            if Error_Results[0][i] <= Error_Results[3][i]*1.2:
                col = 'cornflowerblue'
            else:
                col = 'maroon'
            ax2.scatter(Error_Results[0][i], Error_Results[3][i], c=col)
        
        ax2.set_title('Fitting Error of Predicted Circuit vs. True Circuit Fitting ')
        ax2.grid(visible=True)
        ax2.legend([points[0], points[9]], ['Prediciotn is better/similar', 'Prediciotn is worse'])
        ax2.set_xlabel('Fitting Error of Predicted Circuit [%]')
        ax2.set_ylabel('Fitting Error of Ground Truth Circuit[%]')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        fig.tight_layout(pad=8.0)
        fig.savefig('Evaluation results/Top1Predscatter.png', transparent=False, facecolor='white')
        fig.close()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 13))
        topk_error = Error_Results[:3].min(axis=0)
        points = []
        for i in range(len(Error_Results[0])):
            if topk_error[i] <= Error_Results[4][i]*1.2:
                col = 'cornflowerblue'
                # label = 'Prediciotn is better'
            else: 
                col = 'maroon'
                # label='Prediciotn is worse'
            point = ax1.scatter(topk_error[i], Error_Results[4][i], c=col)
            points.append(point)
        ax1.set_title('Fitting Error of Top-3 Predicted Circuit vs. Universal Circuit Fitting ')
        ax1.grid(visible=True)
        ax1.legend([points[0], points[2]], ['Prediciotn is better', 'Prediciotn is worse'])
        ax1.set_xlabel('Fitting Error of Top-3 Predicted Circuit [%]')
        ax1.set_ylabel('Fitting Error of Universal Circuit [%]')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # plt.savefig('PredvsU.png', transparent=False,facecolor='white')
        # plt.show()
        points=[]
        for i in range(len(Error_Results[0])):
            if topk_error[i]<=Error_Results[3][i]*1.2:
                col = 'cornflowerblue'
            else:
                col='maroon'
            point=ax2.scatter(topk_error[i], Error_Results[3][i], c=col)
            points.append(point)
        ax2.set_title('Fitting Error of Top-3 Predicted Circuit vs. True Circuit Fitting ')
        ax2.grid(visible=True)
        ax2.legend([points[0], points[2]], ['Prediciotn is better/similar', 'Prediciotn is worse'])
        ax2.set_xlabel('Fitting Error of Predicted Circuit [%]')
        ax2.set_ylabel('Fitting Error of Ground Truth Circuit[%]')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        fig.tight_layout(pad=8.0)
        fig.savefig('Evaluation results/Top3Predscatter.png', transparent=False, facecolor='white')
        fig.close()

        plt.figure(figsize=(9, 9))
        sns.histplot(Time_results[0], kde=False)
        sns.histplot(Time_results[-2], kde=False)
        sns.histplot(Time_results[:3].sum(axis=0), kde=False, alpha=0.3)
        plt.title('Distribuation of Fitting Time')
        plt.xlabel('Time')
        plt.legend(['Top-1 Prediction', 'Gound Truth', 'Top-3 Prediction'])
        plt.close()

        plt.figure(figsize=(9, 9))
        plt.hist(Time_results[0], bins=50, alpha=0.5, color='coral')
        plt.hist(Time_results[-2], bins=50, alpha=0.5, color='teal')
        plt.title('Histogram of Fitting Time')
        plt.xlabel('Time')
        plt.legend(['Top-1 Prediction', 'Gound Truth', 'Top-3 Prediction'])
        plt.grid()
        plt.savefig('Evaluation results/TimeDist.png', transparent=False, facecolor='white')
        plt.close()
