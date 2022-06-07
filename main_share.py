#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:  
Data:     2022-06-017
Author:   Jean-Hughes Fournier L. 
Model:    Physics-Guided Recurrent Data-Driven model (PGRDD) 
Paper:    A physics-guided recurrent data-driven model for long-time prediction 
          of nonlinear partial differential equations, Journal of Computational Physics, 2022

"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from datetime import datetime
import pickle
import os
import glob
from sklearn.metrics import mean_squared_error
# Machine learning libraries
import torch
from torch import nn


plt.close('all')

#######################################################################################################################
# 0. Parameters:
#######################################################################################################################

# Dataset:
path_material = "utils/Material_properties/data_spline/"
dataset_path = "data_preprocessed/2022-05-28_01-37_100Simus_3D/"   # Heat imposed in supra layer, hastelloy 500um (set Q_scale=503)

# Use T^(n-2) and T^(n-1) to predict T^(n). Previous: n-2, Current: n-1, Predict: n
predictors = ['T_supra_tn_2_xi(K)', 'T_supra_tn_1_xi(K)', 'Q_supra_tn_1_xi(W/m3)'] 
target     = ['T_supra_tn_xi(K)']
variables  = ['xi(m)'] + ['tn(s)'] + predictors + target 
tag        = {'x':'xi(m)', 't':'tn(s)' }
 
# ML model parameters:
P = {
    'steps_init'           : 2,           # (2)        Initial number of recurrent steps during training 
    'tolerance'            : 5e-5,        # (5e-5)     When train_loss reach 'tolerance' -> 'steps' = 'steps' + 'steps_increment'
    'steps_increment'      : 1,           # (1)        Steps increase by a amount of 'steps_increment' every 'epochs_before_new_step' epochs
    'steps_max'            : 2,           # (2)        Maximum number of epoch before increase steps if tolerance not reached
    'lambda_L2'            : 0,           # (0)        L2 regularizer 
    'batch_size'           : 3000,        # (3000)     Mini-batch size
    'l_r'                  : 1e-3,        # (1e-3)     Learning rate
    'epochs'               : 4000,        # (4000)     Number of epochs
     
    'CONV1_init'           : [1,0,1],     # [1,0,1]    Convolution kernel (K_A in the paper)
    'CONV1_scale'          : 0.5,         # (0.5)      Scaling factor of convollutional kernel
    
    'CONV2_init'           : [-1,0,1],    # [-1,0,1]   Convolution kernel (K_B in the paper)
    'CONV2_scale'          : 0.5,         # (0.5)      Scaling factor of convollutional kernel
    
    'k_eff_scale'          : 2e3,         # (2e3)   
    'k_eff_dh'             : [1,20,20,1], # [1,20,20,1]

    'rhomCp_eff_scale'     : 7e7,         # (7e7)      
    'rhomCp_eff_dh'        : [1,20,20,1], # [1,20,20,1]

    'hconv_scale'          : 7e6,         # (7e6)
    'h_conv_dh'            : [1,20,20,1], # [1,20,20,1]
    
    'Q_scale'              : 503          # (503)
    }

P_PDE = {
    'h_silver_top'     : 2e-6,   # Thickness of top layer in meter
    'h_supra'          : 1e-6,   # Thickness middle layer in meter
    'h_hast'           : 500e-6, # Thickness of bottom layer in meter
    'w'                : 1e-2,   # Width of the tape in meter
    'L'                : 1e-2,   # Length of tape in meter
    'dt'               : 1e-4,   # Time step  in second
    'T_amb'            : 77      # Coolant temperature in Kelvin
}
########################################################################################################################


# Main class
class PINN(nn.Module):
    def __init__(self, P, P_PDE, labels):
        super().__init__()
        
        # Save parameters inside the class 
        self.P = P
        self.P_PDE = P_PDE
        
        # Define neural net for k_eff(T)
        self.net_keff = self.neural_net( P['k_eff_dh'] ) # k_eff = net(T)
        # Define neural net for rhomCeff(T)
        self.net_rhomCpeff = self.neural_net( P['rhomCp_eff_dh'] ) 
        # Define neural net for h_conv(T)
        self.net_hconv = self.neural_net( P['h_conv_dh'] ) 
            
        # Define Convolutional layers CONV1
        self.CONV1 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel1 = np.array( P['CONV1_scale'] * np.array(P['CONV1_init']) ) #
        self.CONV1.weight.data = torch.tensor( kernel1, dtype=torch.float ).unsqueeze(0).unsqueeze(0)

        
        # Define Convolutional layers CONV2
        self.CONV2 = nn.Conv1d( in_channels=1, out_channels=1, kernel_size=3, bias=False)
        kernel2 = np.array( P['CONV2_scale'] * np.array(P['CONV2_init']) ) #
        self.CONV2.weight.data = torch.tensor( kernel2, dtype=torch.float ).unsqueeze(0).unsqueeze(0)

        # Define optimizer:
        self.optimizer = torch.optim.Adam( self.parameters(), lr=P['l_r'], weight_decay= P['lambda_L2'], amsgrad=False)
    
    def neural_net(self, layers):
        # Generate a standard Neural Network architecture
        seq = []
        for i in range(len(layers)-2):
            seq.append( ('Linear'+str(i), nn.Linear(layers[i], layers[i+1])) )
            seq.append( ('Relu'+str(i), nn.ReLU()) )
        seq.append( ('Linear'+str(i+1), nn.Linear(layers[i+1], layers[i+2])) )   
        # Create neural network
        model_seq = torch.nn.Sequential( OrderedDict( seq ) )
        model_seq.apply(init_weights)
        return model_seq
   
    def scale_T(self, T):
        # Purpose: scale temperature before entering the NN
        return ( T - self.P_PDE['T_min']) / (self.P_PDE['T_max'] - self.P_PDE['T_min'])
    
    def inv_scale_T(self, T):
        # Purpose: unscale temperature before entering the NN
        return  (self.P_PDE['T_max'] - self.P_PDE['T_min']) * T + self.P_PDE['T_min']
    
    def forward_keff(self, T): # 
        # Forward pass in  NN_k
        # T is unscaled
        T_sc   = self.scale_T(T)
        out_sc = self.net_keff( T_sc ) 
        out    = self.P['k_eff_scale'] * out_sc 
        return out.flatten()
    
    def forward_dkeffdT(self, k, T): 
        # Compute dk/dT using automatic differentiation
        dummy  = torch.ones( len(T),1 ).flatten() # 
        out = torch.autograd.grad( k, T, grad_outputs=dummy, create_graph=True, only_inputs=True)[0]
        return out
    
    def forward_rhomCpeff(self, T): 
        # Forward pass in  NN_C
        # T is unscaled
        T_sc   = self.scale_T(T)
        out_sc = self.net_rhomCpeff( T_sc ) 
        out    = self.P['rhomCp_eff_scale'] * out_sc
        return out.flatten()
    
    def forward_hconv(self, T): 
        # Forward pass in  NN_h
        # T is unscaled
        T_sc   = self.scale_T(T)
        out_sc = self.net_hconv( T_sc ) 
        out    = self.P['hconv_scale'] * out_sc 
        #########################################################
        # Not allow negative values
        out = torch.maximum(torch.zeros(out.size()),out)
        #########################################################
        
        return out.flatten()
    
    def forward(self, T_current, T_previous, Q): # T:(mini-batch, Nx)
        # Add boundary condition
        bound = self.P_PDE['T_amb'] * torch.ones((T_current.shape[0], 1)) # (mini-batch,1)
                
        # Convolution layers
        conv1_T = self.CONV1(T_current.unsqueeze(dim=1)).squeeze(dim=1) / P['CONV1_scale']  # 
        conv2_T = self.CONV2(T_current.unsqueeze(dim=1)).squeeze(dim=1) / P['CONV2_scale']  # 
        
        # Shape size
        a = T_current[:,1:-1].shape[0]
        b = T_current[:,1:-1].shape[1]
        T_in = T_current[:,1:-1].flatten().unsqueeze(1) # size( a*b, 1 )
        
        # Material properties:
        keff_T      = self.forward_keff( T_in )            # NN_k(T)
        dkeffdT     = self.forward_dkeffdT( keff_T, T_in ) # d NN_k(T)/dT, 
        rhomCpeff_T = self.forward_rhomCpeff( T_in )       # NN_C(T)
        
        # Convection coefficient  
        h_convec_T  = self.forward_hconv(T_in) # NN_h
        
        # Uncomment those lines to use true values of h_conv
        # spl_h_conv  = pickle.load( open(path_material + "spline_h_conv.pkl", "rb" ))
        # h_convec_T  = spl_h_conv( T_in.detach() - self.P_PDE['T_amb'] ) # h(T-T_amb)  (detach() is used here)
        # h_convec_T  = torch.tensor( h_convec_T, dtype=torch.float )
            
        # Bring back original size
        rhomCpeff_T = rhomCpeff_T.reshape(a,b)
        keff_T      = keff_T.reshape(a,b)
        dkeffdT     = dkeffdT.reshape(a,b)
        h_convec_T  = h_convec_T.reshape(a,b)
        
        # Temperature-dependent coefficients
        gamma_T = 2 * self.P_PDE['dt'] / ( rhomCpeff_T )
        A_T = 2 * keff_T * P_PDE['dt'] / ( rhomCpeff_T * self.P_PDE['dx']**2 )
        B_T = dkeffdT * P_PDE['dt'] / (2 * rhomCpeff_T * self.P_PDE['dx']**2 )
        
        # Geometry coefficient for the  convection
        h_tot = P_PDE['h_silver_top']+P_PDE['h_supra']+P_PDE['h_hast'] # total tape thickness
        S     = 2 * P_PDE['L'] * ( h_tot + P_PDE['w'] )                # Surface where convection occurs
        V     = h_tot * P_PDE['w'] * P_PDE['L']                        # Volume of the tape
        
        ######################################################################################
        # Dufort-Frankel scheme (nonlinear heat equation)
        out =  ( A_T*conv1_T + B_T*conv2_T**2 + 
                 (1-A_T)*T_previous[:,1:-1] + 
                  gamma_T * (Q[:,1:-1] - S/V * h_convec_T * (T_current[:,1:-1]-self.P_PDE['T_amb'])) 
                ) / (1+A_T) 
        ######################################################################################
        # Add boundary conditions
        out = torch.cat( ( out, bound ), dim=1) # 
        out = torch.cat( ( bound, out ), dim=1) # 
        
        return out
        
    
    def fit(self, cube_train, cube_val, DATA, epochs):# Inputs: CUBE
        # Mini batch value
        K = self.P['batch_size']
        steps = self.P['steps_init']
        
        # Loop over epoch
        Loss_train_epoch, Loss_val_epoch = [], []
        STEPS = []
        for epo in range(int(epochs)):
            # Convert Cubes [i,n,s] into Sequences [N, L, H_in], N: samples, L: time sequence, H_in: temperature along x-axis
            Seq_train = cubes2sequences(cube_train, steps)
            Seq_val   = cubes2sequences(cube_val, steps)
                    
            # Convert to Torch tensors 
            Seq_train_Tn_2 = torch.tensor( Seq_train['T_supra_tn_2_xi(K)'],    dtype=torch.float, requires_grad=True) 
            Seq_train_Tn_1 = torch.tensor( Seq_train['T_supra_tn_1_xi(K)'],    dtype=torch.float, requires_grad=True) 
            Seq_train_Q_1  = torch.tensor( Seq_train['Q_supra_tn_1_xi(W/m3)'], dtype=torch.float) 
            Seq_train_Tn   = torch.tensor( Seq_train['T_supra_tn_xi(K)'],      dtype=torch.float) 
            
            Seq_val_Tn_2 = torch.tensor( Seq_val['T_supra_tn_2_xi(K)'],    dtype=torch.float, requires_grad=True) 
            Seq_val_Tn_1 = torch.tensor( Seq_val['T_supra_tn_1_xi(K)'],    dtype=torch.float, requires_grad=True) 
            Seq_val_Q_1  = torch.tensor( Seq_val['Q_supra_tn_1_xi(W/m3)'], dtype=torch.float) 
            Seq_val_Tn   = torch.tensor( Seq_val['T_supra_tn_xi(K)'],      dtype=torch.float) 
            
            # Number of samples
            N = Seq_train_Tn.size()[0]
            
            # Loop over mini-batches (K: mini-batch, N: batch)
            loss_train_batch, loss_val_batch = [], []
            permutation = torch.randperm(Seq_train_Tn.size()[0])
            for k in range( 0, N, K):
                # Define minibatch
                indices = permutation[k:k+K]
                batch_Seq_train_Tn_2 = Seq_train_Tn_2[indices]
                batch_Seq_train_Tn_1 = Seq_train_Tn_1[indices]
                batch_Seq_train_Q_1  = Seq_train_Q_1[indices]
                batch_Seq_train_Tn   = Seq_train_Tn[indices]
                 
                # Initialize state at a starting point of a sequence
                Tn_recur_previous_train = batch_Seq_train_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_train  = batch_Seq_train_Tn_1[:,0,:] # (scaled) size: (K,Nx)
                
                Tn_recur_previous_val = Seq_val_Tn_2[:,0,:] # (scaled) size: (K,Nx)
                Tn_recur_current_val  = Seq_val_Tn_1[:,0,:] # (scaled) size: (K,Nx)
               
                # Loop over multi-step predictions ('steps' time steps)
                Tn_pred_train_list, Tn_true_train_list, Tn_pred_val_list, Tn_true_val_list = [],[],[],[]
                for n in range( 0, int(np.floor(steps-1)) ): # 
                    # [N, L, H_in]
                    Q_train       = batch_Seq_train_Q_1[:,n,:] # size: (K,Nx)
                    Tn_true_train = batch_Seq_train_Tn[:,n,:]  # (unscaled) size:(K,Nx)
                    
                    Q_val         = Seq_val_Q_1[:,n,:] #
                    Tn_true_val   = Seq_val_Tn[:,n,:]  # 
                    
                    # Predictions
                    Tn_predict_train = self.forward( Tn_recur_current_train, Tn_recur_previous_train, Q_train ) # size: (K, 2*Nx)
                    Tn_predict_val   = self.forward( Tn_recur_current_val,   Tn_recur_previous_val, Q_val )
                    
                    # Save for loss function
                    Tn_pred_train_list.append( Tn_predict_train ) # Accumulate a sequence of predictions
                    Tn_true_train_list.append( Tn_true_train )
                    
                    Tn_pred_val_list.append( Tn_predict_val ) # Accumulate a sequence of predictions
                    Tn_true_val_list.append( Tn_true_val )
                    
                    # Predictions become inputs
                    Tn_recur_previous_train = Tn_recur_current_train.clone().float()
                    Tn_recur_current_train  = Tn_predict_train.clone().float()
                    
                    Tn_recur_previous_val = Tn_recur_current_val.clone().float()
                    Tn_recur_current_val  = Tn_predict_val.clone().float()
                
                # Error  (K*L,Nx) (for one mini-batch)
                difference_Tn_train = torch.stack(Tn_pred_train_list,dim=1).reshape( len(indices)*(steps-1), Nx ) - torch.stack(Tn_true_train_list,dim=1).reshape( len(indices)*(steps-1), Nx )
                difference_Tn_val   = torch.stack(Tn_pred_val_list,dim=1).reshape( len(Seq_val_Tn_1)*(steps-1), Nx ) - torch.stack(Tn_true_val_list,dim=1).reshape( len(Seq_val_Tn_1)*(steps-1), Nx )
            
                # Mean square error for one mini-batch
                loss_train = torch.mean(torch.square( difference_Tn_train )) #
                loss_val   = torch.mean(torch.square( difference_Tn_val )) #
                
                #####################################################################################################
                # Compute gradient and update weights
                #####################################################################################################
                self.optimizer.zero_grad()
                loss_train.backward(retain_graph=True)
                self.optimizer.step()
                #####################################################################################################
                
                # Save loss per batch
                loss_train_batch.append( loss_train.item() ) # 
                loss_val_batch.append( loss_val.item() ) 
                
            # Save loss every epoch
            Loss_train_epoch.append( sum(loss_train_batch)/( len(loss_train_batch)) ) # Average of losses on all mini-batch
            Loss_val_epoch.append( sum(loss_val_batch)/( len(loss_val_batch)) )
            STEPS.append( steps ) # Accumulate steps every epoch for display
            
            # Predict material properties:
            self.keff_77      = self.forward_keff( torch.tensor([[77]]))
            self.rhomCpeff_77 = self.forward_rhomCpeff( torch.tensor([[77]]))
            self.hconv_77 = self.forward_hconv( torch.tensor([[77]]))
                
            print( 'Epoch:%03i, steps:%i, train: %.2e, val: %.2e, k(77): %.1f, C(77): %.1e, h(77): %.1e' % (epo+1, steps, Loss_train_epoch[-1],Loss_val_epoch[-1], self.keff_77, self.rhomCpeff_77,self.hconv_77))

            # Increase "steps" when error reach a certain tolerance
            if (Loss_train_epoch[-1] <= self.P['tolerance']) and (steps < P['steps_max']):
                steps = steps + self.P['steps_increment']

        return Loss_train_epoch, Loss_val_epoch, STEPS

    def predict(self, T_current, T_previous, Q_current): # Unscaled input
        # Convert to Torch tensors
        T_current  = torch.tensor( T_current, dtype=torch.float, requires_grad=True )
        T_previous = torch.tensor( T_previous, dtype=torch.float )
        Q_current  = torch.tensor( Q_current, dtype=torch.float ) 
        
        # Prediction (Finite difference for time derivative)
        Tn_predict = self.forward( T_current, T_previous, Q_current )# size: 
        return Tn_predict
    
    def set_device(self, device):
        self.device = device
        
def Multistep_predictions(cubes_DATA, simu_index, Ns):
    ## Test set Multi-step Prediction at each time(test set)
    Y_pred_list, Y_pred_recur_list, Y_true_list, tn_list = [],[],[],[]
    
    # Data shape [Nx,Nt,Ns]
    X1     = cubes_DATA['T_supra_tn_2_xi(K)'] 
    X2     = cubes_DATA['T_supra_tn_1_xi(K)']
    X3     = cubes_DATA['Q_supra_tn_1_xi(W/m3)']
    Y_true = cubes_DATA['T_supra_tn_xi(K)']
    tn     = cubes_DATA[tag['t']]
    
    Nx, Nt, Ns = X1.shape[0], X1.shape[1], X1.shape[2]
    
    # Initialize values at t=0
    previous_recur = np.expand_dims( X1[:,0,simu_index], 1).T
    current_recur  = np.expand_dims( X2[:,0,simu_index], 1).T
    
    for n in range( 0, Nt ): # n=0 -> t=0.002s (first prediction is done at 0.002s)
        # Prediction
        Y_pred_recur = model.predict( current_recur, previous_recur, np.expand_dims( X3[:,n,simu_index],1 ).T ).detach().numpy()
        # Save data
        Y_pred_recur_list.append( Y_pred_recur )
        Y_true_list.append( np.expand_dims(Y_true[:,n,simu_index], 1).T )
        tn_list.append( tn[0,n,simu_index]  )
        # Recurrence
        previous_recur = np.copy(current_recur)
        current_recur  = np.copy(Y_pred_recur)
        
    # Gather results
    out = {}
    out['time'] = np.array( tn_list )
    out['pred'] = np.squeeze( np.stack(Y_pred_recur_list),axis=1)
    out['true'] = np.squeeze( np.array( Y_true_list), axis=1)
    return out

def func_keff(T,P_PDE):
    h_tot = P_PDE['h_silver_top'] + P_PDE['h_supra'] + P_PDE['h_hast']
    k_eff = (P_PDE['h_silver_top']*P_PDE['k_Ag_spline'](T) + 
             P_PDE['h_supra']*P_PDE['k_ybco_spline'](T)    + 
             P_PDE['h_hast']*P_PDE['k_hast_spline'](T)) / h_tot
    return k_eff

def func_dkeffdT(T,P_PDE):
    h_tot    = P_PDE['h_silver_top'] + P_PDE['h_supra'] + P_PDE['h_hast']
    dkdT_eff = (P_PDE['h_silver_top']*P_PDE['dkdT_Ag_spline'](T) + 
                P_PDE['h_supra']*P_PDE['dkdT_ybco_spline'](T)    + 
                P_PDE['h_hast']*P_PDE['dkdT_hast_spline'](T)) / h_tot
    return dkdT_eff
    
def func_rhomCpeff(T, P_PDE):
    h_tot      = P_PDE['h_silver_top'] + P_PDE['h_supra'] + P_PDE['h_hast']
    rhomCp_eff = ( P_PDE['h_silver_top']*P_PDE['rhom_Ag_cte']*P_PDE['Cp_Ag_spline'](T) + 
                   P_PDE['h_supra']*P_PDE['rhom_ybco_cte']*P_PDE['Cp_ybco_spline'](T)  +
                   P_PDE['h_hast']*P_PDE['rhom_hast_cte']*P_PDE['Cp_hast_spline'](T)  ) / h_tot
    return rhomCp_eff

def load_data(path):
    # ****************************************************
    # Load dataset  from many csv files (concatenate all simulations)
    #       path: path of the folder containing the results of the COMSOL simulations (.csv files)
    #       labels: features to be load
    # Output: dataframe
    # ****************************************************
    # List of path of .csv files
    files = [f for f in glob.glob(path)]
    files.sort()
    
    X = []
    # Loop over COMSOL simulations (.csv files) 
    for file in files:
        df = pd.read_csv(file)
        # Remove empty spaces in header
        df.columns = [c.replace(' ', '') for c in df.columns]
        # Add unique_index column
        Simu_id = int(file.split('/')[-1].split('.')[-2])
        df['Simu_index'] = Simu_id
        X.append( df ) # List of dataframe
        print( 'Simu_id: %i'%(Simu_id) )
    return pd.concat(X) # Output: Dataframe

def DF_to_CUBE(DATA):
    # Purpose: Dataframe to cube [i,n,s], i: position index, time index, simulation index
    # Get the different COMSOL simulations unique identifier
    simu_unique = sorted( set( DATA['Simu_index'].to_numpy()) )
    tn_unique   = sorted( set( DATA['tn(s)'].to_numpy()) )
    # Create CUBE of data
    cube_lists = [ [] for n in range(len(DATA.columns)) ]
    for s in simu_unique: # 
        seq_lists = [ [] for n in range(len(DATA.columns)) ]
        for tn in tn_unique: # 
            crit = np.array( (DATA['Simu_index'] == s) & (DATA['tn(s)'] == tn) ) 
            for k in range(len(DATA.columns)):
                seq_lists[k].append( DATA[crit].sort_values('xi(m)')[DATA.columns[k]].values )
        for k in range(len(DATA.columns)):
            cube_lists[k].append( np.vstack( seq_lists[k] ).T )
    out = {}
    for k in range(len(DATA.columns)):
        out[DATA.columns[k]] = np.stack( cube_lists[k],2 ) 
    return out

def cubes2sequences(cube_DATA, steps):
        # Input: cube_DATA: dictionnary of cube of DATA 
        # The axis of the cubes are [i,n,s] i: index along "x", n: index along time, s: index along simulations
        # Output: dictionnary of sequences
        # Each sequence has the dimension [batch, sequence along time, Nx] or [N, L, Nx]
        tn = np.stack( sorted( set( cube_DATA['tn(s)'].flatten()) ) )
        
        # Loop over element in dictionnary
        sequences = {}
        for element in cube_DATA.keys():
            # Loop over starting points for multi-step predictions (tn)
            temp = []
            for i_t, t in enumerate( tn[:len(tn)-int(np.floor(steps))] ): 
                temp.append( cube_DATA[element][ :, i_t:(i_t+steps+1), : ] )
            sequences[element] = np.transpose( np.concatenate(temp,axis=2), (2,1,0) )
        return sequences
    
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_ (0.01)
        
def RMSE(true, predict, scale='lin'):
    if scale.lower() == 'lin':
        return np.sqrt(mean_squared_error( true, predict ))
    if scale.lower() == 'log':
        temp =  ( np.log10(predict.to_numpy()) - np.log10(true.to_numpy()))**2
        return np.sqrt( temp.sum()/len(temp) )


if __name__ == "__main__":
    
    ########################################################################################################################
    # 1. Import data
    ########################################################################################################################
    # Load train set
    DATA_train = load_data( dataset_path + "train/*.csv" ) # 
    # Load test set
    DATA_test  = load_data( dataset_path + "test/*.csv" ) 
    # Load validation set
    DATA_val   = load_data( dataset_path + "val/*.csv" ) 
    
    # Get the different COMSOL simulations unique identifier
    simu_unique_test = sorted( set( DATA_test['Simu_index'].to_numpy()) )
    tn_unique = sorted( set( DATA_train['tn(s)'].to_numpy()) )
    xi_unique = sorted( set( DATA_train['xi(m)'].to_numpy()) )
    
    
    Nt = len(tn_unique) # Number of points along time
    Nx = len(xi_unique) # Number of points along time
    Ns_test = len(simu_unique_test)
    
    # Material properties
    P_PDE['k_Ag_spline']      = pickle.load( open( path_material + "spline_k_Ag.pkl",    "rb" ) ) 
    P_PDE['dkdT_Ag_spline']   = pickle.load( open( path_material + "spline_dkdT_Ag.pkl", "rb" ) )
    P_PDE['Cp_Ag_spline']     = pickle.load( open( path_material + "spline_Cp_Ag.pkl",   "rb" ) )
    
    P_PDE['k_ybco_spline']    = pickle.load( open( path_material + "spline_k_ybco.pkl",    "rb" ) )
    P_PDE['dkdT_ybco_spline'] = pickle.load( open( path_material + "spline_dkdT_ybco.pkl", "rb" ) )
    P_PDE['Cp_ybco_spline']   = pickle.load( open( path_material + "spline_Cp_ybco.pkl",   "rb" ) )

    P_PDE['k_hast_spline']    = pickle.load( open( path_material + "spline_k_hast.pkl",    "rb" ) )
    P_PDE['dkdT_hast_spline'] = pickle.load( open( path_material + "spline_dkdT_hast.pkl", "rb" ) )
    P_PDE['Cp_hast_spline']   = pickle.load( open( path_material + "spline_Cp_hast.pkl",   "rb" ) )
    
    P_PDE['rhom_Ag_cte']      = 10490 # kg/m3
    P_PDE['rhom_ybco_cte']    = 6390  # kg/m3
    P_PDE['rhom_hast_cte']    = 8890  # kg/m3
    
    # Creation of new variables 
    P_PDE['T_min'] = DATA_train[target[-1]].min()
    P_PDE['T_max'] = DATA_train[target[-1]].max()
    P_PDE['Q_min'] = DATA_train['Q_supra_tn_1_xi(W/m3)'].min()
    P_PDE['Q_max'] = DATA_train['Q_supra_tn_1_xi(W/m3)'].max()
    P_PDE['L']     = DATA_train[tag['x']].max()
    P_PDE['t_end'] = DATA_train[tag['t']].max()
    P_PDE['Nx']    = len(xi_unique) 
    P_PDE['Nt']    = len(tn_unique) 
    P_PDE['dx']    = P_PDE['L']/(P_PDE['Nx']-1)
    P_PDE['k_eff_77']      = func_keff( 77, P_PDE )
    P_PDE['rhomCp_eff_77'] = func_rhomCpeff( 77, P_PDE )
    
    
    # Convert DataFrame (DF) of data to "CUBE" of data. Indexes: [i,n,s] == [space_index, time_index, simu_index ]
    print("Converting DF to cube...")
    cubes_DATA_train = DF_to_CUBE( DATA_train)
    cubes_DATA_test  = DF_to_CUBE( DATA_test)
    cubes_DATA_val   = DF_to_CUBE( DATA_val)
    print("Conversion done")
    
    ########################################################################################################################
    # 2. Train model
    ########################################################################################################################

    # Data-time
    now = datetime.now()
    now_str = '%i-%02i-%02i_%02i-%02i-%02i' % ( now.year, now.month, now.day, now.hour, now.minute, now.second )
    
    # Create folder
    os.mkdir('results/'+ now_str ) 
    
    # Parameters for training
    # Training model
    model = PINN( P, P_PDE, labels= predictors+target )
    Loss_train_epoch, Loss_val_epoch, STEPS = model.fit( cubes_DATA_train, cubes_DATA_val, DATA_train, P['epochs'] )
    
    ########################################################################################################################
    # 3. Predictions:
    ########################################################################################################################
    print("Make predictions:")
    
    # Data shape [Nx,Nt,Ns]
    PRED_test, TRUE_test = [],[]
    for s in range(Ns_test):
        out = Multistep_predictions(cubes_DATA_test, s, Ns_test)
        PRED_test.append( out['pred'] )
        TRUE_test.append( out['true'] )
    
    time_test = out['time'] 
    PRED_test = np.stack(PRED_test).T
    TRUE_test = np.stack(TRUE_test).T
    
    RMSE_test = []
    for n in range(PRED_test.shape[1]):
        RMSE_test.append( RMSE( TRUE_test[:,n,:],PRED_test[:,n,:] )  )
        
    RMSE_test = np.stack(RMSE_test)
    
    
    ########################################################################################################################
    # 4. Display results:
    ########################################################################################################################
    print("Display results:")
    
    # 1) Display training error over epochs
    plt.figure(figsize=(4,6))
    plt.subplot(2,1,1)
    plt.semilogy( 1+np.arange(len(Loss_train_epoch)) * P['epochs']/len(Loss_train_epoch), Loss_train_epoch, color = 'blue', label='Train' )
    plt.semilogy( 1+np.arange(len(Loss_val_epoch))   * P['epochs']/len(Loss_val_epoch),   Loss_val_epoch,':', color = 'orange', label='Val' )
    plt.grid()
    plt.title(now_str)
    plt.legend()
    plt.tight_layout()
    plt.ylabel('MSE')
    plt.subplot(2,1,2)
    plt.plot( np.arange(len(STEPS)) * P['epochs']/(len(STEPS)-1), STEPS, label='' )
    plt.grid()
    plt.legend()
    plt.ylabel('Multi-Steps training')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('results/'+now_str+'/Fig1_Training.png', dpi=300)
    
    
    # 2) Plot PREDICTIONS from test set (multi-step prediction)
    Tmax_index = np.unravel_index(cubes_DATA_test['T_supra_tn_xi(K)'].argmax(), cubes_DATA_test['T_supra_tn_xi(K)'].shape )
    choice_simu = 0 #Tmax_index[2]
    plt.figure(figsize=(6,5))
    plt.plot( 1000*np.array(xi_unique), np.squeeze(TRUE_test[   :,len(tn_unique)-1,choice_simu]),'k',   label='t(n)=%.1f ms (Dataset)'  % (1e3*tn_unique[len(tn_unique)-1]) )
    plt.plot( 1000*np.array(xi_unique), np.squeeze(PRED_test[   :,len(tn_unique)-1,choice_simu]),'b:',  label='t(n)=%.1f ms (ML model)' % (1e3*tn_unique[len(tn_unique)-1]) )
    plt.plot( 1000*np.array(xi_unique), np.squeeze(TRUE_test[   :,48,choice_simu]),'k',   label='t(n)=%.1f ms (Dataset)'  % (1e3*tn_unique[48]) )
    plt.plot( 1000*np.array(xi_unique), np.squeeze(PRED_test[   :,48,choice_simu]),'b:',  label='t(n)=%.1f ms (ML model)' % (1e3*tn_unique[48]) )
    plt.plot( 1000*np.array(xi_unique), np.squeeze(TRUE_test[   :,0,choice_simu]),'k',   label='t(n)=%.1f ms (Dataset)'  % (1e3*tn_unique[0]) )
    plt.plot( 1000*np.array(xi_unique), np.squeeze(PRED_test[   :,0,choice_simu]),'b:',  label='t(n)=%.1f ms (ML model)' % (1e3*tn_unique[0]) )
    plt.grid()
    plt.legend()
    plt.ylabel('T(K)')
    plt.xlabel('x(mm)')
    plt.title(now_str)
    plt.tight_layout()
    plt.savefig('results/'+now_str+'/Fig2_Predictions.png', dpi=300)
    
    # 3) Material properties
    x_max = 400
    x = np.linspace(1,400,160)
    xx = np.linspace(1,400,1000)
    xx_tensor = torch.tensor( xx, dtype=torch.float, requires_grad=True ).unsqueeze(1)
    keff_pred    = model.forward_keff(xx_tensor)
    dkeffdT_pred = model.forward_dkeffdT(keff_pred,xx_tensor).detach()
    rhomCpeff    = model.forward_rhomCpeff(xx_tensor).detach()
    h_conv       = model.forward_hconv(xx_tensor).detach()
    
    plt.figure(figsize=(8,7))
    # Subplot 1
    plt.subplot(2,2,1)
    plt.plot( x, func_keff( x, P_PDE ),'o', label='k (true)' )
    plt.plot( xx, keff_pred.detach(), label='k (pred)' )
    plt.plot( [P_PDE['T_min'],P_PDE['T_min']], [0,keff_pred.detach().max()], ':k' )
    plt.plot( [P_PDE['T_max'],P_PDE['T_max']], [0,keff_pred.detach().max()], ':k' )
    plt.legend()
    plt.grid()
    #plt.axis([20,x_max,0,2000])
    plt.title(now_str)
    plt.ylabel('Heat conductivity (W.m$^{-1}$.K$^{-1}$)')
    
    # Subplot 2
    plt.subplot(2,2,2)
    plt.plot( x, func_rhomCpeff( x, P_PDE ),'o', label='C (true)' )
    plt.plot( xx, rhomCpeff.detach(), label='C (pred)' )
    plt.plot( [P_PDE['T_min'],P_PDE['T_min']], [0,rhomCpeff.detach().max()], ':k' )
    plt.plot( [P_PDE['T_max'],P_PDE['T_max']], [0,rhomCpeff.detach().max()], ':k' )
    plt.legend()
    plt.grid()
    #plt.axis([20,x_max,0,4e8])
    plt.tight_layout()
    plt.xlabel('T(K)')
    plt.ylabel('Heat Capacity (J.m$^{-3}$.K$^{-1}$)')
    
    # Subplot 3
    plt.subplot(2,2,3)
    spl_h_conv  = pickle.load( open( path_material + "spline_h_conv.pkl", "rb" ))
    h_convec_T  = spl_h_conv( x - P_PDE['T_amb'] ) # h(T-T_amb)  
    plt.plot( x, h_convec_T, label='h_conv (true)' )
    plt.plot( xx, h_conv.detach(), label='h_conv (pred)' )
    plt.plot( [P_PDE['T_min'],P_PDE['T_min']], [0,h_conv.detach().max()], ':k' )
    plt.plot( [P_PDE['T_max'],P_PDE['T_max']], [0,h_conv.detach().max()], ':k' )
    plt.legend()
    plt.grid()
    #plt.axis([20,x_max,0,1e5])
    plt.tight_layout()
    plt.xlabel('T(K)')
    plt.ylabel('Convection coefficient (W.m$^{-2}$.K$^{-1}$)')
    plt.savefig( 'results/' + now_str + '/Fig3_keff_rhomCpeff.png', dpi=300)
    
