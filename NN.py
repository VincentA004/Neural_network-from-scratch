import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class NeuralNetwork:
    
    def __init__(self,network_architecture,learning_rate):
        self.structure = network_architecture
        self.alpha = learning_rate
        self.generate_weights_and_biases()
        
    def generate_weights_and_biases(self):
        layer_index = 1
        self.weight_matrix = []
        self.bias_matrix = []
        while layer_index < len(self.structure):
            input_node = self.structure[layer_index-1]
            output_node = self.structure[layer_index]
            weights = .01*np.random.randn(output_node,input_node)
            bias = np.zeros((output_node,1))
            self.weight_matrix.append(weights)
            self.bias_matrix.append(bias)
            layer_index+=1
            
    
    def forward(self,input):
        number_of_layers = len(self.structure)    
        layer_index = 0
        input_vector = np.array(input,ndmin=2).T
        self.forward_cache_vector = [input_vector]
        while layer_index < number_of_layers-1:
            current_input_vector = self.forward_cache_vector[-1]
            w = np.dot(self.weight_matrix[layer_index],current_input_vector) 
            z = w+self.bias_matrix[layer_index]
            if layer_index == number_of_layers - 1:
                output_vector = self.softmax(z)
            else:
                output_vector = self.activation_reLU_forward(z)
            self.forward_cache_vector.append(output_vector)
            layer_index += 1
    
    def backward(self,one_hot_y):
        number_of_layers = len(self.structure) 
        layer_index = number_of_layers -1
        target_vector = np.array(one_hot_y,ndmin=2).T
        true_loss = self.loss_catagorical_crossentropy(self.forward_cache_vector[-1],target_vector)
        loss = self.output_backward(self.forward_cache_vector[-1],target_vector)
        nth_error = loss / len(self.forward_cache_vector[-1])
        while layer_index > 0:
            output_vector = self.forward_cache_vector[layer_index]
            input_vector =  self.forward_cache_vector[layer_index-1]
            nth_layer_gradient = self.activation_reLU_backward(nth_error, output_vector)  
            self.gradient_decent(input_vector, nth_layer_gradient,layer_index)
            nth_error = np.dot(self.weight_matrix[layer_index-1].T,nth_error)
            layer_index -=1
        return true_loss
    
    
    
    def activation_reLU_backward(self, d_val,input_vector):
        reLU_d_inputs = d_val.copy()
        reLU_d_inputs[input_vector <= 0]=0
        return reLU_d_inputs   
    
    def gradient_decent(self,input_vector,nth_layer_gradient,layer_index):
        nth_weight_gradient = np.dot(nth_layer_gradient, input_vector.T)
        nth_bias_gradient = np.sum(nth_layer_gradient)
        self.weight_matrix[layer_index-1] -= self.alpha * nth_weight_gradient
        self.bias_matrix[layer_index-1] -= self.alpha * nth_bias_gradient
        
    def softmax(self,input_vector):
        numerator_shift = np.exp(input_vector - np.max(input_vector))
        predicted_probabilities = numerator_shift / np.sum(numerator_shift)
        output_vector = predicted_probabilities
        return output_vector
    

    def output_backward(self,predicted_probabilities,one_hot_y):
        last_layer_gradient =  predicted_probabilities - one_hot_y 
        return last_layer_gradient
    
         
    def activation_reLU_forward(self,input):
        output = np.maximum(0,input)
        return output
    
    def activation_reLU_backward(self, d_val,output_vector):
        reLU_d_inputs = d_val.copy()
        reLU_d_inputs[output_vector <= 0]=0
        return reLU_d_inputs
    
    def activation_tanh_forward(self,input):
        output = np.tanh(input)
        return output
    
    def activation_tanh_backward(self,d_val,output_vector):
        tanh_d_inputs = d_val*(1 - (output_vector)**2)
        
    def loss_catagorical_crossentropy(self,y_hat, one_hot_y):
        epsilon = 1.0e-7
        y_hat_shifted = np.clip(y_hat, epsilon,1-epsilon)
        correct_confidences = np.sum(y_hat_shifted*one_hot_y, axis = 1)
        return -np.log(correct_confidences)
        
            
    def train_single(self,input,one_hot_y):
        self.forward(input) 
        return self.backward(one_hot_y)     
        
    def train(self,data,labels,epochs):
        for e in range(epochs):
            for i in range(len(data)):
                l = self.train_single(data[i],labels[i])
                if i % 1000 == 0:
                    print("Epochs: ", e+1,"Iteration: ", i, " loss: ", np.mean(l))
                    #print('predicted_probabilities:',self.forward_cache_vector[-1])
                
        print('training done!')        
                
    def single_pass(self,input):
        self.forward(input)
        return self.forward_cache_vector[-1]
        
    
    def evaluate_multiple(self,data,labels):
        correct,wrong = (0,0)
        for i in range(len(data)):
            ans = self.single_pass(data[i])
            ans_max = ans.argmax()
            if ans_max == labels[i]:
                correct+=1
            else:
                wrong+=1
        return correct, wrong
    
    def evaluate_single(self,data,label):
        result = ''
        i = random.randint(0,10000)
        ans = self.single_pass(data[i])
        ans_max = ans.argmax()
        if ans_max == label[i]: 
            result = f'The digit predicted: {ans_max}: \u2705'
        else:
            result = f'The digit predicted: {ans_max}: \u274c'
        return result,i
    
     def run_test(self,data,label):
        result, i = self.evaluate_single(data, label)
        img = data[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        print(plt.show())
        print(result)
        
               

def image_preprocessor(image_dimensions, number_of_unique_labels, dataset_path):
    number_of_total_pixels = image_dimensions**2
    
    data_train = pd.read_csv(dataset_path + '\\' + 'mnist_train.csv')
    data_test = pd.read_csv(dataset_path + '\\' + 'mnist_test.csv')
    
    train_data = np.array(data_train)
    test_data = np.array(data_test)
    
    factor = 0.99 / 255
    
    train_images = np.asfarray(train_data[:,1:]) * factor + 0.01
    test_images = np.asfarray(test_data[:,1:]) * factor + 0.01
    
    train_labels = np.asfarray(train_data[:,:1])
    test_labels = np.asfarray(test_data[:,:1])
    
    label_range = np.arange(number_of_unique_labels)
    train_labels_one_hot = (label_range==train_labels).astype(float)
    test_labels_one_hot = (label_range==test_labels).astype(float)
    
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99
    
    return train_images, test_images, train_labels_one_hot, train_labels, test_labels_one_hot , test_labels, number_of_total_pixels
    
    
    
    

    
    
    
    
    
        
            
        
    
