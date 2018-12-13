import cv2
import numpy as np
from math import sqrt,degrees,sqrt
from random import shuffle
import os

#Global variables
all_feature_vector=[]
test_feature_vector=[]
expected_outputs=[]
alpha=0.01
gweights=[]
gbias=[]

#Calculate Horizontal and vertical gradients
def Gradient_operator(image):
    #Caclulating Gx using Prewitt's operator
    x_mask=[[-1,0,1],
            [-1,0,1],
            [-1,0,1]]
    x_mask=np.asarray(x_mask)
    #Initialize gx elements to 0
    N=len(image)
    M=len(image[0])
    gx = np.zeros((N, M))
    
    #Apply prewitt mask to calculate gx.
    for i in range(1,N-1):
        for j in range(1,M-1):
            gx[i][j]=image[i-1][j-1]*-1+image[i-1][j+1]+image[i][j-1]*-1+image[i][j+1]+image[i+1][j-1]*-1+image[i+1][j+1]
    
    #Calculating Gy using Prewitt's operator
    y_mask=[[1,1,1],
            [0,0,0],
            [-1,-1,-1]]
    y_mask=np.asarray(y_mask)
    #Initialize gy elements to 0
    gy = np.zeros((N, M))
    #Apply prewitt mask to calculate gy
    for i in range(1,N-1):
        for j in range(1,M-1):
            gy[i][j]=image[i-1][j-1]+image[i-1][j]+image[i-1][j+1]+image[i+1][j-1]*-1+image[i+1][j]*-1+image[i+1][j+1]*-1
    
    #Normalize the gradients
    gx=gx/3.0
    gy=gy/3.0

    return gx,gy

#Calculate Gradient Angles
def Magnitude_Gradient(gx,gy):
    N = len(gx)
    M = len(gx[0])
    Magnitude = np.zeros((N, M))
    Gradient_angle = np.zeros((N, M))
    
    #Calculate the Gradient Magnitude
    for i in range(1,N-1):
        for j in range(1,M-1):
            Magnitude[i][j]=((sqrt(gx[i][j]*gx[i][j]+gy[i][j]*gy[i][j]))/np.sqrt(2))
    
    #Calculate the Gradient Angle
    for i in range(1,N-1):
        for j in range(1,M-1):
            #Gradient angle when horizontal gradient is not equal to 0.
            if(gx[i][j]!=0):
                Gradient_angle[i][j]=(degrees(np.arctan(gy[i][j]/gx[i][j])))
            #Gradient angle when horizontal gradient=0
            elif(gx[i][j]==0):
                if (gy[i][j]<0):
                    Gradient_angle[i][j]=-90
                elif (gy[i][j]>0):
                    Gradient_angle[i][j]=90
                else:
                    Gradient_angle[i][j]=0
                    
            #Converting negative angles to positive and in range [-10,170)
            if(Gradient_angle[i][j]<-10):
                Gradient_angle[i][j]+=360
            if Gradient_angle[i][j]>=170:
                Gradient_angle[i][j]=Gradient_angle[i][j]-180
          
    return Magnitude, Gradient_angle

#Function for assigning proportions of Magnitudes of pixel values to bins based on distance from the bin center.
def cell_bins(magnitude,gradient,all_bins):
    rows=len(magnitude)
    columns=len(magnitude[0])
    for i in range(0,rows,8):
        for j in range(0,columns,8):
            bins=np.zeros(9)
            for r in range(i,i+8):
                for c in range(j,j+8):
                    #Magnitude distribution When the gradient angle is between 0 and -10.
                    if(gradient[r][c]<0):
                        temp=magnitude[r][c]*((0-gradient[r][c])/20)
                        bins[8]+=temp
                        bins[0]+=(magnitude[r][c]-temp)
                    #Magnitude distribution when grdient angle is between 160 and 170.
                    elif gradient[r][c]>160:
                        temp=magnitude[r][c]*((gradient[r][c]-160)/20)
                        bins[0]+=temp
                        bins[8]+=magnitude[r][c]-temp
                    #Magnitude distribution when gradient angle is between 1 and 160.
                    else:
                        if(gradient[r][c]%20==0):
                            bins[(int)(gradient[r][c]/20)]+=magnitude[r][c]
                        elif(gradient[r][c]%20==10):
                            bins[(int)(gradient[r][c]//20)+1]+=0.5*magnitude[r][c]
                            bins[(int)(gradient[r][c]//20)]+=0.5*magnitude[r][c]  
                        else:#(gradient[r][c]%20>10):
                            temp=magnitude[r][c]*((gradient[r][c]-(int)(gradient[r][c]//20)*20)/20)
                            bins[(int)(gradient[r][c]//20)+1]+=temp
                            bins[(int)(gradient[r][c]//20)]+=(magnitude[r][c]-temp)
            all_bins[i//8][j//8]=bins
    return all_bins

#Function to load HOG feature vector for all images in a directory one by one
def LoadImageHistograms(dir1,flag):
    #Scans all images in the directory specified
    for filename in os.listdir(dir1):
        image=cv2.imread(dir1+filename,cv2.IMREAD_COLOR)
        #Extracting the Blue, Green and Red components of the pixel into separate arrays.
        B=image[:,:,0]
        G=image[:,:,1]
        R=image[:,:,2]
        #Computing the grayscale pixel value
        BW_image=0.299*R+0.587*G+0.114*B
        
        #Rounding the grayscale pixel value
        BW_image=np.round(BW_image)
    
        #Calculate gradients
        gx,gy=Gradient_operator(BW_image)
    
        #Calculate magnitude and gradient angle 
        magnitude,gradient_angle=Magnitude_Gradient(gx,gy)
    
        #Saving Gradient magnitude images for test images
        if flag==1:
            cv2.imwrite("C:\\NYU\\CV\\Project2\\magnitude\\"+filename.split('.')[0]+".png", magnitude)

        #Initialize a list to store all the histograms of cells
        all_bins=np.zeros((len(magnitude)//8,len(magnitude[0])//8,9))
    
        #Function call to get histograms of all the cells in a list
        histogram=cell_bins(magnitude,gradient_angle,all_bins)
    
        #Function call to get the HOG feature vector.
        feature_vector=blocks(histogram)
        
        if filename=='crop001278a.bmp':
            np.savetxt("C:\\NYU\\CV\\Project2\\crop001278a.txt", feature_vector, fmt="%s", newline='\n')
        
        if filename=='crop001045b.bmp':
            np.savetxt("C:\\NYU\\CV\\Project2\\crop001045b.txt", feature_vector, fmt="%s", newline='\n')
            
        #Reshape the feature vector from 1 x 7524 to 7524 x 1
        feature_vector=np.array(feature_vector).reshape(-1,1) #7524,1
        
        #Append the HOG feature vector to Training/Testing set respective list
        if flag==0:
            all_feature_vector.append(feature_vector) 
        else:
            test_feature_vector.append(feature_vector)
        
#Form the HOG feature vector from the calculated histograms for cells.
def blocks(histogram):
    result_vector=[]
    for i in range(len(histogram)-1):
        for j in range(len(histogram[0])-1):
            vector=[]
            for p1 in range(i,i+2):
                for p2 in range(j,j+2):
                    vector.extend(histogram[p1][p2])
            vector=np.array(vector)
            #Do L2 Normalization
            vector_square=np.square(vector)
            vector_sum=sum(vector_square)
            #To avoid the value of NaN in feature_vector, simply insert 0
            if(vector_sum==0.0):
                result_vector.extend(np.zeros((36),dtype=float))
            else:
                vector=vector/sqrt(vector_sum)
                result_vector.extend(vector)
    return result_vector

#Randomly initialize the weights and biases for the Neural Network
def CreateNN(hidden_layer_perceptron_count):
    weights=[]
    bias=[]
    np.random.seed(0)
    w1=np.random.randn(hidden_layer_perceptron_count,len(all_feature_vector[0]))*0.01 #250,7524
    weights.append(w1)
    w2=np.random.randn(1,hidden_layer_perceptron_count)*0.01 #1,250
    weights.append(w2)
    b1=np.random.randn(hidden_layer_perceptron_count,1)*0.01 #250,1
    bias.append(b1)
    b2=np.random.randn(1,1)*0.01
    bias.append(b2)   
    return weights,bias                 

#Feedforward of a Neural Network
def RunNeuralNetwork(feature,result,weights,bias):
    z1=np.dot(weights[0],feature)+bias[0] #250,1
    #Apply ReLU filter
    z1=np.maximum(z1,0) #250,1
    z2=np.dot(weights[1],z1)+bias[1] #1,1
    #Apply Sigmoid filter
    z2=1/(1+np.exp(-z2)) #1,1
    #Calculate error
    error=0.5*((result-z2)**2)
    #Now update weights
    weights,bias=UpdateWeights(z1,z2,feature,result,weights,bias)
    return error,weights,bias

#Function to update weights, biases
def UpdateWeights(z1,z2,feature,result,weights,bias):
    temp2=(z2-result)*z2*(1-z2)
    dw2=np.dot(temp2,z1.T)
    db2 = np.sum(temp2,axis=1, keepdims=True)
    temp1=np.dot(weights[1].T,temp2)*ReLuDerivation(z1)
    dw1=np.dot(temp1,feature.T)
    db1 = np.sum(temp1,axis=1, keepdims=True)
    #Update weights
    weights[1]=weights[1]-alpha*dw2
    weights[0]=weights[0]-alpha*dw1
    #Update biases
    bias[1]=bias[1]-alpha*db2
    bias[0]=bias[0]-alpha*db1
    return weights,bias
    
#Derivative of ReLU
def ReLuDerivation(x):
    return 1. * (x > 0)

#Function to Backpropogate till the error_cost becomes 0.01, ie, till 1% error cost
def Backpropogation(weights,bias):
    cost=100
    epoch=0
    while True:
        epoch=epoch+1
        if(cost>0.01):
            error=0
            for i in range(len(expected_outputs)):
                err,weights,bias=RunNeuralNetwork(all_feature_vector[i],expected_outputs[i],weights,bias)
                error+=err
            cost=error/20
        else:
            print('Total Epochs ',epoch)
            return weights,bias
    return weights,bias
    
#Calculate output for the testing images by passing it's HOG feature vector through trained Neural network
def Test(feature,weights,bias):
    z1=np.dot(weights[0],feature)+bias[0] #250,1
    z1=np.maximum(z1,0) #250,1
    z2=np.dot(weights[1],z1)+bias[1] #1,1
    z2=1/(1+np.exp(-z2))
    if(z2>=0.5):
        print(z2,'  Human detected')
    else:
        print(z2,'  No human')

def main():
    #Training image directories
    dir1='C:\\NYU\\CV\\Project2\\Images\\Human\\Train_Positive\\'
    dir2='C:\\NYU\\CV\\Project2\\Images\\Human\\Train_Negative\\'
    
    #Load Histograms for the images in above directories
    LoadImageHistograms(dir1,0)
    LoadImageHistograms(dir2,0)    
    
    #Expected Result is set as 1 for Positive training images and 0 for negative training images
    y=[1 for i in range(10)]+[0 for i in range(10)]
    
    #Zip feature_vector with it's expected output
    feature_output_zip=list(zip(all_feature_vector,y))
    
    #Shuffle the training set to mix positive, negative training examples
    shuffle(feature_output_zip)
    
    #Now, unzip the shuffled training set
    feature_output_zip,y=zip(*feature_output_zip)
    
    for i in range(len(y)):
        expected_outputs.append(y[i])
        all_feature_vector[i]=feature_output_zip[i]
        
    #number of perceptrons in hidden layer
    hidden_layer_count=250
    
    #Function call to initialize weights, biases
    weights,bias=CreateNN(hidden_layer_count)
    
    #Function call to Backpropogate
    weights,bias=Backpropogation(weights,bias)
    gweights.extend(weights)
    gbias.extend(bias)

    #Run Test images on the trained Neural network
    test_dir='C:\\NYU\\CV\\Project2\\Images\\Human\\Test_Positive\\'
    LoadImageHistograms(test_dir,1)
    test_dir='C:\\NYU\\CV\\Project2\\Images\\Human\\Test_Neg\\'
    LoadImageHistograms(test_dir,1)
    for i in range(len(test_feature_vector)):
        Test(test_feature_vector[i],weights,bias)
    
    #Save weights and biases of trained model
    np.savetxt("w1.txt", gweights[0], fmt="%s")
    np.savetxt("w2.txt", gweights[1], fmt="%s")
    np.savetxt("b1.txt", gbias[0], fmt="%s")
    np.savetxt("b2.txt", gbias[1], fmt="%s")
    
if __name__=="__main__":
    main()