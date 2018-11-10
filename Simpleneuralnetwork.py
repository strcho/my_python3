# coding: utf-8

import numpy
import scipy.special
import time

class neuralNetwork: 
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.win = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x:scipy.special.expit(x)
        
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin = 2).T
        hidden_inputs = numpy.dot(self.win,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        #反向传播误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        self.win += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        hidden_inputs = numpy.dot(self.win,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        
#输入层节点
input_nodes = 784
#隐藏层节点
hidden_nodes = 300
#输出层节点
output_nodes = 10
#学习率
learning_rate = 0.1

#训练次数
epoche = 3


n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

with open("mnist_dataset/mnist_train.csv",'r') as training_data_file:
    training_data_list = training_data_file.readlines()

start_time = time.clock()
for e in range(epoche):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/225.0*0.99)+0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
end_time = time.clock()
time_num = (end_time-start_time)/60.0


with open("mnist_dataset/mnist_test.csv",'r') as test_data_file:
    test_data_list = test_data_file.readlines()

scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    
    inputs = (numpy.asfarray(all_values[1:])/225.0*0.99)+0.01
    outputs = n.query(inputs)
    lable = numpy.argmax(outputs)
    print(correct_label,"correct lable",lable,"network's answer")
    if (lable == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
scorecard_array = numpy.asarray(scorecard)
print("正确率：",scorecard_array.sum()/scorecard_array.size*100,"%","\n训练用时：%.3f分钟。"%time_num)

