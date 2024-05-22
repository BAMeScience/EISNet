from generate_data import generate_data
from Data_processing import pre_process
from train import train
from predict import Prediction


class main():
    def __init__(self):
        """
        define parameters for training the neural network
        """
        self.paramters = {
            'Data Size': int(1e3),
            'save_data': False,
            'Frequency_lb': -3,
            'Frequency_ub': 8,
            'Samples': 200,
            'Scaling': True,
            'Scaling Type': ['stand','minmax'],
            'Device': 'cuda:2',
            'model': 'net1',
            'Start Training': False,
            'continue_training': False,
            'batch_size': 512,
            'learning_rate': 1e-4,
            'epochs': 100
        }

        assert not ((self.paramters['continue_training'] == True) and
                    (self.paramters['Start Training'] == True)), \
        'Choose eather a new training or continue Training '

    def generate_data(self):
        """
        function for generating EIS data from the universal circuit (see paper)
        """
        GD = generate_data(size=self.paramters['Data Size'],
                            w_lb=self.paramters['Frequency_lb'],
                            w_ub=self.paramters['Frequency_ub'],
                            samples=self.paramters['Samples'])
        return GD.generate()
    
    def preprocess_data(self, data=None):
        """
        function for preparing and standardizing the data for the training the testing
        """
        PP = pre_process(scale=self.paramters['Scaling'],
                         scale_type=self.paramters['Scaling Type'][0],
                         device=self.paramters['Device'],
                         New_Training=self.paramters['Start Training'],
                         Neual_Network=self.paramters['model'],
                         data=data)

        return PP.process()

    def Train(self, Training_data):
        """
        Calls the training module
        """
        Training = train(Training_data=Training_data,
                         device=self.paramters['Device'],
                         continue_training=self.paramters['continue_training'],
                         batch_size=self.paramters['batch_size'],
                         learning_rate=self.paramters['learning_rate'],
                         epochs=self.paramters['epochs'],
                         Neual_Network=self.paramters['model'])
        Training.start_training()

    def Predict(self, Test_data):
        """
        Calls the testing module
        """
        Pred = Prediction(Test_Data=Test_data,
                          device=self.paramters['Device'],
                          Neual_Network=self.paramters['model'])
        Pred.predict()


if __name__ == '__main__':
    model = main()
    data = model.generate_data()
    Training_data, Test_data, (Z_real_test_wide, Z_imag_test_wide) = model.preprocess_data(data=data)
    #model.Train(Training_data)
    model.Predict(Test_data)