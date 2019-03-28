import math
import random
import numpy as np
from random import seed
class neural_network:
    def __init__(self,pattern):

        self.AI,self.AH,self.AO,self.WIH,self.WHO=[],[],[],[],[]
        self.ni=3
        self.nh=3
        self.no=1

        self.AI=[1.0]*(self.ni)
        self.AH=[1.0]*(self.nh)
        self.AO=[1.0]*(self.no)
        
        #seed(23)
        for i in range(self.ni):
            self.WIH.append([0.0]*self.nh)

        for i in range(self.nh):
            self.WHO.append([0.0]*self.no)
        """
        self.WIH=[[0.15,0.25],[0.20,0.30]]
        self.WHO=[[0.40,0.50],[0.45,0.55]]
        """    
        for i in range(self.ni):
            for j in range(self.nh):
                self.WIH[i][j]=random.uniform(-0.2,0.2)

        for i in range(self.nh):
            self.WHO.append([0.0]*self.no)

        for i in range(self.nh):
            for j in range(self.no):
                self.WHO[i][j]=random.uniform(-2.0,2.0)
        self.CIH=[]
        self.CHO=[]
        for i in range(self.ni):
            self.CIH.append([0.0]*self.nh)

        for j in range(self.nh):
            self.CHO.append([0.0]*self.no)
            
        #print(self.WIH)
        #print(self.WHO)
    def backprop(self,inputs,expected,output,learning_rate,mom):
        output_delta=[0.0]*self.no
        for k in range(self.no):
            error=expected[k]-output[k]
            output_delta[k]=error*self.AO[k]*(1-self.AO[k])
        for j in range(self.nh):
            for k in range(self.no):
                delta_weight_output=output_delta[k]*self.AH[j]
                self.WHO[j][k]+=(mom*self.CHO[j][k])+(learning_rate)*delta_weight_output
                self.CHO[j][k]=delta_weight_output
        hidden_delta=[0.0]*self.nh
        for j in range(self.nh):
            erorr=0.0
            for k in range(self.no):
                error+=self.WHO[j][k]*output_delta[k]
            hidden_delta[j]=error*self.AH[j]*(1-self.AH[j])
            

        for i in range(self.ni):
            for j in range(self.nh):
                delta_weight_hidden=hidden_delta[j]*self.AI[i]
                self.WIH[i][j]+=(mom*self.CIH[i][j])+(learning_rate)*delta_weight_hidden
                self.CIH[i][j]=delta_weight_hidden
    def forwardprop(self,inputs):
            for i in range(self.ni-1):
                self.AI[i]=inputs[i]

                for j in range (self.nh):
                    sum=0.0
                    for i in range(self.ni):
                        sum+=self.AI[i]*self.WIH[i][j]

                    self.AH[j]=sigmoid(sum)
            #print("ah\n")
            #print(self.AH)
            for k in range(self.no):
                sum=0.0
                for j in range(self.nh):
                    sum+=self.AH[j]*self.WHO[j][k]

                self.AO[k]=sigmoid(sum)
                
            return self.AO

    def print_training(self,pattern,overall_loss):
        for p in pattern:
            inputs=p[0]
            print("inputs:",p[0]," outputs:",self.forwardprop(inputs)," target:",p[1],"loss:",overall_loss)
    def print_test(self,pat):
          for p in pat:
            inpu=p[0]
            print("inputs:",p[0]," outputs:",self.forwardprop(inpu)," target:",p[1])
        
    def checking_test(self,pat):
        for p in pat:
            inpu=p[0]
            self.forwardprop(inpu)
        self.print_test(pat)
    def train(self,pattern):
                loss=[]
                loss=[0.0]*self.no
                for i in range(100):
                    for p in pattern:
                        inputs=p[0]
                        output=self.forwardprop(inputs)
                        expected=p[1]
                        #print("epoch:",i,"\n")
                        self.backprop(inputs,expected,output,learning_rate=0.1,mom=0.5)
                        #print("wih:",self.WIH,"\n")
                        #print("who:\n",self.WHO,"\n")
                        loss=[(0.5*((a - b)**2))for a,b in zip(output,expected)]
                        overall_loss=np.sum(loss)
                    self.print_training(pattern,overall_loss)
                print("training done!!")

def sigmoid(x):
    return 1/(1+math.exp(-x))

def main():
    pat=[
         [[2.7810836,2.550537003],[0]],
         [[7.673756466,3.508563011],[1]]
         ]
    
    pattern=[
	[[1.465489372,2.362125076],[0]],
	[[3.396561688,4.400293529],[0]],
	[[1.38807019,1.850220317],[0]],
	[[3.06407232,3.005305973],[0]],
	[[7.627531214,2.759262235],[1]],
	[[5.332441248,2.088626775],[1]],
	[[6.922596716,1.77106367],[1]],
	[[8.675418651,-0.242068655],[1]]
	]

    """[
        [[0.05,0.10],[0.01,0.99]]
        ]
    """
    """
    pattern=[
        [[0,0],[0]],
        [[0,1],[1]],
        [[1,0],[1]],
        [[1,1],[1]]
        ]
    """
    neural=neural_network(pattern)
    neural.train(pattern)
    neural.checking_test(pat)

if __name__=="__main__":
    main()
