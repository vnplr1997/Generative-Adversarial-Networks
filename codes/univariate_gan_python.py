import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical #for one hot encoding

def sigmoid(x):
    return(1/(1+np.exp(-x)))
    
def sigmoiddash(a):
    return a*(1-a)

def relu(x):

    x[x<0]=0
    return x

def reludash(x):

    x[x>0]=1
    x[x<0]=0
    return x

def gen_layers_size(Z,X,h):
    n_x=np.shape(Z)[0]
    n_h=h
    n_y=np.shape(X)[0]
    return(n_x,n_h,n_y)


def dis_layers_size(X,ou,h):
    n_x=np.shape(X)[0]
    n_h=h
    n_y=ou
    return(n_x,n_h,n_y)
    
    
def init_gen_params(n_x,n_h,n_y):

    GW1=np.random.randn(n_h,n_x)*0.01
    Gb1=np.zeros((n_h,1))
    GW2=np.random.randn(n_y,n_h)*0.01
    Gb2=np.zeros((n_y,1))

    gen_params={'GW1':GW1,'Gb1':Gb1,'GW2':GW2,'Gb2':Gb2}
    return gen_params


def init_dis_params(n_x,n_h,n_y):

    DW1=np.random.randn(n_h,n_x)*0.01
    Db1=np.zeros((n_h,1))
    DW2=np.random.randn(n_y,n_h)*0.01
    Db2=np.zeros((n_y,1))

    dis_params={'DW1':DW1,'Db1':Db1,'DW2':DW2,'Db2':Db2}
    return dis_params


def gen_feed_forward(Gen_params,Z):

    W1=Gen_params['GW1']
    b1=Gen_params['Gb1']
    W2=Gen_params['GW2']
    b2=Gen_params['Gb2']

    Z1=np.dot(W1,Z).reshape(np.shape(b1))+b1
    A1=relu(Z1)
    Z2=np.dot(W2,A1).reshape(np.shape(b2))+b2
    A2=relu(Z2)
    gen_cache={'Z1':Z1,'Z2':Z2,'A1':A1,'A2':A2}
    
    return A2,gen_cache


def dis_feed_forward(Dis_params,X):

    W3=Dis_params['DW1']
    b3=Dis_params['Db1']
    W4=Dis_params['DW2']
    b4=Dis_params['Db2']

    Z3=np.dot(W3,X).reshape(np.shape(b3))+b3
    A3=relu(Z3)
    Z4=np.dot(W4,A3).reshape(np.shape(b4))+b4
    A4=sigmoid(Z4)
    dis_cache={'Z3':Z3,'Z4':Z4,'A3':A3,'A4':A4}
    
    return A4,dis_cache

def dis_backprop(dis_params,dis_cache,Y,X):
    
    W3=dis_params['DW1']
    b3=dis_params['Db1']
    W4=dis_params['DW2']
    b4=dis_params['Db2']

    Z3=dis_cache['Z3']
    A3=dis_cache['A3']
    Z4=dis_cache['Z4']
    A4=dis_cache['A4']
   
    dA4=A4-Y.reshape(np.shape(A4))
    dW4=dA4*A3.T
    db4=dA4
    
    r3=reludash(Z3)
    local=r3*W4.T

    dW3=dA4*local*X.T
    db3=dA4*local
    

    dis_dparams={'dDW1':dW3,'dDb1':db3,'dDW2':dW4,'dDb2':db4}
    return dis_dparams

def update_dis_params(dis_params,dis_dparams,lr):

    W3=dis_params['DW1']
    b3=dis_params['Db1']
    W4=dis_params['DW2']
    b4=dis_params['Db2']

    dW3=dis_dparams['dDW1']
    db3=dis_dparams['dDb1']
    dW4=dis_dparams['dDW2']
    db4=dis_dparams['dDb2']

    dis_params['DW1']=W3-(lr)*dW3
    dis_params['Db1']=b3-(lr)*db3.reshape(np.shape(b3))
    dis_params['DW2']=W4-(lr)*dW4
    dis_params['Db2']=b4-(lr)*db4.reshape(np.shape(b4))

    return dis_params


def gen_predict(gen_params,Z,n):
    P=np.zeros((2,n))
    for j in range(n):
        A2,cache=gen_feed_forward(gen_params,Z[:,j])  
        P[:,j]=A2.T  
    
    return P

def dis_predict(dis_params,X):
    P=np.random.rand(np.shape(X)[1])
    for j in range(np.shape(X)[1]):
        A2,cache=dis_feed_forward(dis_params,X[:,j])
        p=1 if A2>0.5 else 0
        P[j]=p   
    
    return P


def gen_backprop(dis_params,dis_cache,gen_params,gen_cache,Y,X):
    

    W3=dis_params['DW1']
    b3=dis_params['Db1']
    W4=dis_params['DW2']
    b4=dis_params['Db2']

    Z3=dis_cache['Z3']
    A3=dis_cache['A3']
    Z4=dis_cache['Z4']
    A4=dis_cache['A4']

    
    W1=gen_params['GW1']
    b1=gen_params['Gb1']
    W2=gen_params['GW2']
    b2=gen_params['Gb2']

    Z1=gen_cache['Z1']
    A1=gen_cache['A1']
    Z2=gen_cache['Z2']
    A2=gen_cache['A2']
   
    dZ4=A4-Y.reshape(np.shape(A4))
    dW4=dZ4*A3.T
    db4=dZ4
    
    r3=reludash(Z3)
    local=r3*W4.T

    dW3=dZ4*local*A2.T 
    db3=dZ4*local

    r2=reludash(Z2)
    local2=r2*W3.T

    dW2=dZ4*local*local2*A1.T
    db2=dZ4*local*local2

    r1=reludash(Z1)
    local3=r1*W2.T

    dW1=dZ4*local*local2*local3*X.T
    db1=dZ4*local*local2*local3
    

    gen_dparams={'dGW1':dW1,'dGb1':db1,'dGW2':dW2,'dGb2':db2}
    return gen_dparams 

def update_gen_params(gen_params,gen_dparams,lr):

    W1=gen_params['GW1']
    b1=gen_params['Gb1']
    W2=gen_params['GW2']
    b2=gen_params['Gb2']

    dW1=gen_dparams['dGW1']
    db1=gen_dparams['dGb1']
    dW2=gen_dparams['dGW2']
    db2=gen_dparams['dGb2']

    gen_params['GW1']=W1-(lr)*dW1
    gen_params['Gb1']=b1-(lr)*db1.reshape(np.shape(b1))
    gen_params['GW2']=W2-(lr)*dW2
    gen_params['Gb2']=b2-(lr)*db2.reshape(np.shape(b2))

    return gen_params

def real_samples(n):
    
    X1 = 2*np.pi*np.random.rand(n) - np.pi
    X2 = np.sin(X1)
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = np.hstack((X1, X2)).T
    y = np.ones((n, 1)).T
    return X, y

def latent_points(latent_dim, n):
    
    x_input = np.random.randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim).T
    return x_input

def fake_samples(gen_params, latent_dim, n):

    Z = latent_points(latent_dim, n)
    X = gen_predict(gen_params,Z,n)
    y = np.zeros((n, 1)).T
    return X, y





def main():

    n=100
    latent_dim=5
    lr=0.1
    
    x_real,y_real=real_samples(n)
    Z=latent_points(latent_dim,n)

    n_1,n_2,n_3=gen_layers_size(Z,x_real,25)
    n_4,n_5,n_6=dis_layers_size(x_real,1,15)
    
    gen_params=init_gen_params(n_1,n_2,n_3)
    dis_params=init_dis_params(n_4,n_5,n_6)
    

    for i in range(1):
        
        x_real,y_real=real_samples(n)
        real=[x_real[0,:],x_real[1,:]]
        
        Z=latent_points(latent_dim,n)

        x_fake,y_fake=fake_samples(gen_params, latent_dim, n)
        fake=[x_fake[0,:],x_fake[1,:]]

        X_dis=np.concatenate((real,fake),axis=1)
        y_dis=np.hstack((y_real,y_fake))

        for j in range(np.shape(X_dis)[1]):

            A4,dis_cache=dis_feed_forward(dis_params,X_dis[:,j])

            dis_dparams=dis_backprop(dis_params,dis_cache,y_dis[:,j],X_dis[:,j])
            dW1=dis_dparams['dDW1']
            db1=dis_dparams['dDb1']
            dW2=dis_dparams['dDW2']
            db2=dis_dparams['dDb2']

            dis_params=update_dis_params(dis_params,dis_dparams,lr)
            

        x_gan=latent_points(latent_dim,n)
        y_gan=np.ones((1,n))

        for j in range(np.shape(x_gan)[1]):

            A2,gen_cache=gen_feed_forward(gen_params,x_gan[:,j])
            A4,dis_cache=dis_feed_forward(dis_params,X_dis[:,j])
            
            gen_dparams=gen_backprop(dis_params,dis_cache,gen_params,gen_cache,y_gan[:,j],x_gan[:,j])
            gen_params=update_gen_params(gen_params,gen_dparams,lr)





        

    print(dis_params)
        
    x_real,y_real=real_samples(n)

    p=dis_predict(dis_params,x_real)


    print(y_real)
    print(p)

    Z=latent_points(latent_dim,n)

    x_fake,y_fake=fake_samples(gen_params, latent_dim, n)

    p=dis_predict(dis_params,x_fake)
    print(y_fake)
    print(p)
    

  
