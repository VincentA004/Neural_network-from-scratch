import time
import NN



image_dimensions = 28
number_of_unique_labels = 10
dataset_path = 'C:\\Users\\Vince\\Desktop\\code\\Neural_networks\\Datasets\\Handwriting dataset'
default_config_array = [image_dimensions**2, 100, 30, number_of_unique_labels]
default_learning_rate = 0.01
default_epochs = 3


def main():
    check = True
    config_array = default_config_array
    learning_rate = default_learning_rate
    epochs = default_epochs
    while(check==True):
        option = GUI()
        if(option == 1):
            train_and_test(config_array,learning_rate,epochs)
        elif(option==2):
            learning_rate = float(input("Enter learning rate: "))
            epochs = int(input("Enter epochs: "))
            print('hyper_parameters changed, retrain network to see results')
            time.sleep(2)
        elif(option == 3 ):
            print('\nResetting hyper_parameters to default values')
            config_array = default_config_array
            learning_rate = default_learning_rate
            epochs = default_epochs
            time.sleep(1.5) 
        else:
            check = False
    
        
        
        
        
        
        


def GUI():
    print('\n'+"**************************************\n\tDigit Classifier \n\tBy: Vincent Allam\n**************************************"+'\n')
    print("1. Train model")
    print("2. Change hyper_parameters")
    print("3. Reset hyper_parameters to default values")
    print("4. Quit")
    option_decider = int(input())
    return option_decider
   

def initialize_neural_network(config_array, lr):
    neural_network = NN.NeuralNetwork(config_array, lr)
    return neural_network
           
def train_and_test(config_array,lr,epochs):
    print('\nInitializing neural network...\n')
    neural_network = initialize_neural_network(config_array, lr)
    print('Preprocessing image...\n')
    X_train, X_test, y_train_one_hot, y_train, y_test_one_hot, y_test, image_pixels = NN.image_preprocessor(image_dimensions, number_of_unique_labels, dataset_path)
    print('Training started...')
    t0 = time.time()
    neural_network.train(X_train, y_train_one_hot,epochs)
    t1 = time.time()
    c,w = neural_network.evaluate_multiple(X_train, y_train)
    t2 = time.time()
    accuracy = (c / (c + w))*100
    print("\n Training data summary stats: ")
    print(f'training accuracy: {accuracy}%')
    print(f'total training time: {t1-t0}')
    print(f'total evaluation time: {t2-t1}')
    test_option = 1
    while(test_option != 3):
        print('\n*********************Test model menu*********************\n')
        print('1. Test model: ')
        print('2. Get test accuracy:')
        print('3. quit testing: ')
        test_option = int(input())
        if test_option == 1:
            neural_network.run_test(X_test, y_test)
        elif test_option == 2:
            t1 = time.time()
            c,w = neural_network.evaluate_multiple(X_test, y_test)
            t2 = time.time()
            accuracy = (c / (c + w))*100
            print("\nTesting data summary stats: ")
            print(f'testing accuracy: {accuracy}%')
            print(f'total evaluation time: {t2-t1}')
        else:
            pass
            
            
        
        
    
    
    
                   
main()