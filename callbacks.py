from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import scipy.ndimage
import time
import datetime
import os
import cv2
from skimage.measure import regionprops
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from shutil import copyfile
import tensorflow as tf
import model.unets as unets
import functools
print = functools.partial(print, flush=True)


# Dice Coefficient for 3D Volumes
def dice_3d(predictions, labels):
    epsilon     =  1e-7
    dice_num    =  np.sum(predictions[labels==1])*2.0 
    dice_denom  =  np.sum(predictions) + np.sum(labels)
    return ((dice_num+epsilon)/(dice_denom+epsilon)).astype(np.float32)


# Export Weights Every N Epochs
class WeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, model, min_epoch, weights_num_epochs, weights_dir, init_epoch=0, weights_overwrite=True):
        self.model   = model
        self.N       = weights_num_epochs
        self.M       = min_epoch
        self.D       = weights_dir+'/model_weights.h5'
        self.O       = weights_overwrite
        self.epoch   = init_epoch

    def on_epoch_end(self, epoch, logs={}):

        if ((self.epoch+1)%self.N==0)&(self.epoch!=0)&((self.epoch+1)>=self.M):
            name     =  self.D
            name     =  name.split('.h5')[0] + '_%03d.h5' % (self.epoch+1)
            
            # To Counter {BlockingIOError: Resource temporarily unavailable}
            while True:
                try:
                    tf.keras.models.save_model(self.model, name)
                    print('Model Weights Saved: ', name)
                    break
                except: continue
            if self.O:
                name =  self.D
                name =  name.split('.h5')[0] + '_%03d.h5' % ((self.epoch+1)-self.N)
                
                while True:
                    try:
                        if os.path.exists(name): os.remove(name)
                        break
                    except: continue
        self.epoch  += 1


# Custom Learning Rate Scheduler
class ReduceLR_Schedule(tf.keras.callbacks.Callback):
    """
    Reduce learning rate when model performance has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates.
    """
    def __init__(self, lr_rates, epoch_points):
        self.lr_rates         = lr_rates
        self.epoch_points     = epoch_points
          
    def on_epoch_begin(self, epoch, logs=None):

        assert (len(self.epoch_points)==len(self.lr_rates))

        if ((epoch+1)>=self.epoch_points[0])&((epoch+1)<self.epoch_points[1]): new_lr = self.lr_rates[0]
        if ((epoch+1)>=self.epoch_points[1])&((epoch+1)<self.epoch_points[2]): new_lr = self.lr_rates[1]
        if ((epoch+1)>=self.epoch_points[2])&((epoch+1)<self.epoch_points[3]): new_lr = self.lr_rates[2]
        if ((epoch+1)>=self.epoch_points[3]):                                  new_lr = self.lr_rates[3]
        
        if ((epoch+1)==self.epoch_points[0])|((epoch+1)==self.epoch_points[1])|((epoch+1)==self.epoch_points[2])|((epoch+1)==self.epoch_points[3]):
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print('\nEpoch %03d: ReduceLR_Schedule reducing learning '
                  'rate to %s.' % (epoch+1, new_lr))


# Custom Learning Rate Scheduler
class PolyLR_Schedule(tf.keras.callbacks.Callback):
    """
    Reduce learning rate as per the nn-U-Net training heuristic.
    """
    def __init__(self, initial_lr, exponent, max_epochs):
        self.initial_lr  = initial_lr
        self.exponent    = exponent
        self.max_epochs  = max_epochs
          
    def on_epoch_begin(self, epoch, logs=None):
        new_lr           = self.initial_lr * (1-epoch/self.max_epochs)**self.exponent

        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        print('\nEpoch %03d: PolyLR_Schedule reducing learning '
              'rate to %s.' % (epoch+1, new_lr))


# Cyclic Learning Rate Scheduler
class CyclicLR(tf.keras.callbacks.Callback):
    """
    Instead of monotonically decreasing the learning rate, this method 
    lets the learning rate cyclically vary between reasonable boundary
    values. Training with cyclical learning rates instead of fixed values
    achieves improved classification accuracy without a need to tune and
    often in fewer iterations.

    [1] L.N. Smith (2017), "Cyclical Learning Rates for Training Neural Networks", IEEE WACV
    
    """
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self._reset()
    
    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.clr_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())


# Load Model Weights and Restart/Resume Training
def ResumeTraining(model, weights_dir, resume=True, prefix='model_weights'):
    weights_dir = '/'+prefix+'.h5'
    init_epoch  = 0    

    for f in os.listdir(weights_dir.split(prefix)[0]):
        if (resume) & (weights_dir.split('.h5')[0] in (weights_dir.split(prefix)[0]+f)) & ('.xlsx' not in (weights_dir.split(prefix)[0]+f)):
            temp_epoch = int(((weights_dir.split(prefix)[0]+f).split(weights_dir.split('.h5')[0]+'_')[1]).split('.h5')[0])
            if (temp_epoch > init_epoch): init_epoch = temp_epoch

    for f in os.listdir(weights_dir.split(prefix)[0]):
        if (resume) & (weights_dir.split('.h5')[0] in (weights_dir.split(prefix)[0]+f)) & ('.xlsx' not in (weights_dir.split(prefix)[0]+f)):
            if (init_epoch == int(((weights_dir.split(prefix)[0]+f).split(weights_dir.split('.h5')[0]+'_')[1]).split('.h5')[0])):
   
                print('Loading Model Weights...')
                model = unets.networks.M1.load(path=weights_dir.split(prefix)[0]+f)
                print('Complete: ', weights_dir.split(prefix)[0]+f)
   
    if (init_epoch==0): print('Begin Training @ Epoch ',  init_epoch)
    else:               print('Resume Training @ Epoch ', init_epoch)

    return model, init_epoch


# Evaluation of Patient-Level csPCa Diagnosis
class PCaDetectionValidation(tf.keras.callbacks.Callback):
    def __init__(self, model, generators, min_epoch, every_n_epochs, num_samples, 
                 init_epoch=0, export_metrics=None, probabilistic=False, mc_dropout=False, 
                 prob_iterations=10):

        self.model      = model
        self.train_gen  = generators[0]
        self.valid_gen  = generators[1]
        self.train_ns   = num_samples[0]
        self.valid_ns   = num_samples[1]
        self.min_epoch  = min_epoch
        self.n_epochs   = every_n_epochs
        self.epoch      = init_epoch
        self.save_dir   = export_metrics+'/metrics.xlsx'
        self.proba      = probabilistic  
        self.mc_dropout = mc_dropout
        self.prob_iter  = prob_iterations

    def on_epoch_end(self, epoch, logs={}):
  
        # Every N Epochs
        if ((self.epoch+1)%self.n_epochs==0)&((self.epoch+1)>=self.min_epoch):

            # Initialize Ground-Truth/Prediction Lists + Label Class Counters
            all_train_true, all_train_pred, all_valid_true, all_valid_pred = [],[],[],[]     
            counter_tC0,    counter_tC1,    counter_vC0,    counter_vC1    = 0,0,0,0     

            # Import/Initialize Metric Lists
            while True: 
                try:     # To Counter {BlockingIOError: Resource temporarily unavailable}
                    if (self.epoch!=0)&(os.path.exists(self.save_dir)):
                        xlsx_data                        = pd.read_excel(self.save_dir).values
                        epoch_list                       = list(xlsx_data[:,0])
                        train_auc_list,  valid_auc_list  = list(xlsx_data[:,1]),  list(xlsx_data[:,2])
                        train_pauc_list, valid_pauc_list = list(xlsx_data[:,3]),  list(xlsx_data[:,4])
                        train_sens_list, valid_sens_list = list(xlsx_data[:,5]),  list(xlsx_data[:,6])
                    else:
                        epoch_list                       = []
                        train_auc_list,  valid_auc_list  = [],[]
                        train_pauc_list, valid_pauc_list = [],[]
                        train_sens_list, valid_sens_list = [],[]
                    break
                except: continue

            # Build Detection Model
            detect_model = self.model.get_detect_model()
            
            # Monte Carlo Dropout or Not
            if self.mc_dropout or self.proba: s = self.prob_iter
            else:                             s = 1 
            
            # Training Metrics
            for i in range(self.train_ns):
                patient_case = next(self.train_gen)
                scan         = np.expand_dims(patient_case[0]["image"],     axis=0)      
                label        = np.expand_dims(patient_case[1]["detection"], axis=0)    
                prediction   = np.stack([np.array(detect_model.predict([scan]))[0,:,:,:,1] for _ in range(s)])               

                # Variational Inference-Based Approximation of Predictive Distribution
                prediction, var = prediction.mean(axis=0), prediction.var(axis=0)

                # Append Prediction + Label
                all_train_true.append(label[0,:,:,:,1])
                all_train_pred.append(prediction)

                # Update Class Counters
                if   (int(np.max(np.ceil((label[0,:,:,:,1]))))==0):  counter_tC0 += 1
                elif (int(np.max(np.ceil((label[0,:,:,:,1]))))==1):  counter_tC1 += 1

            # Validation Metrics
            for i in range(self.valid_ns):
                patient_case = next(self.valid_gen)
                scan         = np.expand_dims(patient_case[0]["image"],     axis=0)      
                label        = np.expand_dims(patient_case[1]["detection"], axis=0)    
                prediction   = np.stack([np.array(detect_model.predict([scan]))[0,:,:,:,1] for _ in range(s)])               

                # Variational Inference-Based Approximation of Predictive Distribution
                prediction, var = prediction.mean(axis=0), prediction.var(axis=0)

                # Append Prediction + Label
                all_valid_true.append(label[0,:,:,:,1])
                all_valid_pred.append(prediction)

                # Update Class Counters
                if   (int(np.max((label[0,:,:,:,1])))==0):  counter_vC0 += 1
                elif (int(np.max((label[0,:,:,:,1])))==1):  counter_vC1 += 1
            
            # Calculate Metrics from FROC Pipeline
            train_metrics = perform_FROC_evaluation(y_true={'y': np.array(all_train_true), 'subject_ids': list(range(len(all_train_true)))}, 
                                                    y_pred=np.array(all_train_pred), pre_threshold='dynamic-fast')
            valid_metrics = perform_FROC_evaluation(y_true={'y': np.array(all_valid_true), 'subject_ids': list(range(len(all_valid_true)))}, 
                                                    y_pred=np.array(all_valid_pred), pre_threshold='dynamic-fast')
            train_pauc = train_metrics['pAUC']
            valid_pauc = valid_metrics['pAUC']
            train_auc  = train_metrics['auroc']
            valid_auc  = valid_metrics['auroc']
            train_sens = train_metrics['max_sens']
            valid_sens = valid_metrics['max_sens']
            
            print('-------------------------------------------------------------------------------------------------------------------------')
            print('Patient-Level Validation:')
            print('-------------------------------------------------------------------------------------------------------------------------')
            print('Training AUROC (Benign + GGG 1 [n='+str(counter_tC0)+'] vs GGG 2-5 [n='+str(counter_tC1)+']): ',                 train_auc)
            print('Validation AUROC (Benign + GGG 1 [n='+str(counter_vC0)+'] vs GGG 2-5 [n='+str(counter_vC1)+']): ',               valid_auc)
            print('Training pAUC [0.1-2.5 FPR] (Benign + GGG 1 [n='+str(counter_vC0)+'] vs GGG 2-5 [n='+str(counter_vC1)+']): ',   train_pauc)
            print('Validation pAUC [0.1-2.5 FPR] (Benign + GGG 1 [n='+str(counter_vC0)+'] vs GGG 2-5 [n='+str(counter_vC1)+']): ', valid_pauc)
            print('Training Max Detection Sens. (Benign + GGG 1 [n='+str(counter_vC0)+'] vs GGG 2-5 [n='+str(counter_vC1)+']): ',  train_sens)
            print('Validation Max Detection Sens. (Benign + GGG 1 [n='+str(counter_vC0)+'] vs GGG 2-5 [n='+str(counter_vC1)+']): ',valid_sens)
            print('-------------------------------------------------------------------------------------------------------------------------')
            
            # Update Lists and Clear Counters
            epoch_list.append(self.epoch+1)
            train_auc_list.append(train_auc)
            valid_auc_list.append(valid_auc)
            train_pauc_list.append(train_pauc)
            valid_pauc_list.append(valid_pauc)
            train_sens_list.append(train_sens)
            valid_sens_list.append(valid_sens)

            counter_tC0, counter_tC1 = 0,0    
            counter_vC0, counter_vC1 = 0,0

            # Export Metrics
            metrics = pd.DataFrame(list(zip(epoch_list, train_auc_list,  valid_auc_list,   train_pauc_list,\
                                                        valid_pauc_list, train_sens_list,  valid_sens_list)), 
                                   columns=['epoch',   'train_auroc',   'valid_auroc',    'train_pauc',\
                                                       'valid_pauc',    'train_max_sens', 'valid_max_sens'])
            while True: # To Counter {BlockingIOError: Resource temporarily unavailable}
                try:
                    metrics.to_excel(self.save_dir, encoding='utf-8', index=False)
                    break
                except: continue
        self.epoch += 1


# Evaluation of Patient-Level Prostatic WG/TZ/PZ Segmentation
class AnatomySegmentationValidation(tf.keras.callbacks.Callback):
    def __init__(self, model, generators, min_epoch, every_n_epochs, num_samples, 
                 init_epoch=0, export_metrics=None, probabilistic=False, mc_dropout=False, 
                 prob_iterations=10):

        self.model      = model
        self.train_gen  = generators[0]
        self.valid_gen  = generators[1]
        self.train_ns   = num_samples[0]
        self.valid_ns   = num_samples[1]
        self.min_epoch  = min_epoch
        self.n_epochs   = every_n_epochs
        self.epoch      = init_epoch
        self.save_dir   = export_metrics+'/training_metrics.xlsx'
        self.proba      = probabilistic  
        self.mc_dropout = mc_dropout
        self.prob_iter  = prob_iterations

    def on_epoch_end(self, epoch, logs={}):
        # Every N Epochs
        print("The current LR is {}".format(self.model.optimizer._decayed_lr('float32').numpy()))

        if ((self.epoch+1)%self.n_epochs==0)&((self.epoch+1)>=self.min_epoch):

            # Initialize Lists for Compiling DSC Scores From All Patients/Epoch
            train_dsc_wg, train_dsc_tz, train_dsc_pz = [],[],[]
            valid_dsc_wg, valid_dsc_tz, valid_dsc_pz = [],[],[]

            # Import/Initialize Metric Lists
            while True: 
                try:     # To Counter {BlockingIOError: Resource temporarily unavailable}
                    if (self.epoch!=0)&(os.path.exists(self.save_dir)):
                        xlsx_data                          = pd.read_excel(self.save_dir).values
                        epoch_points                       = list(xlsx_data[:,0])
                        all_train_dsc_wg, all_valid_dsc_wg = list(xlsx_data[:,1]),  list(xlsx_data[:,2]) 
                        all_train_dsc_tz, all_valid_dsc_tz = list(xlsx_data[:,3]),  list(xlsx_data[:,4]) 
                        all_train_dsc_pz, all_valid_dsc_pz = list(xlsx_data[:,5]),  list(xlsx_data[:,6]) 
                    else:
                        epoch_points                       = []
                        all_train_dsc_wg, all_valid_dsc_wg = [],[] 
                        all_train_dsc_tz, all_valid_dsc_tz = [],[] 
                        all_train_dsc_pz, all_valid_dsc_pz = [],[] 
                    break
                except: continue

            # Build Detection Model
            detect_model = self.model.get_detect_model()
            
            # Monte Carlo Dropout or Not
            if self.mc_dropout or self.proba: s = self.prob_iter
            else:                             s = 1 
            
            # Training Metrics
            for i in range(self.train_ns):
                patient_case              = next(self.train_gen)
                scan                      = np.expand_dims(patient_case[0]["image"],     axis=0)      
                label                     = np.expand_dims(patient_case[1]["detection"], axis=0)    
                all_pred                  = np.stack([np.array(detect_model.predict([scan]))[0,:,:,:,:] for _ in range(s)])               
                wg_pred, tz_pred, pz_pred = all_pred[:,:,:,:,0].copy(), all_pred[:,:,:,:,1].copy(), all_pred[:,:,:,:,2].copy()               

                # Variational Inference-Based Approximation of Predictive Distribution
                wg_pred, wg_var  = wg_pred.mean(axis=0), wg_pred.var(axis=0)
                tz_pred, tz_var  = tz_pred.mean(axis=0), tz_pred.var(axis=0)
                pz_pred, pz_var  = pz_pred.mean(axis=0), pz_pred.var(axis=0)

                # Append Predictions + Labels 
                train_dsc_wg.append(dice_3d(1-wg_pred,1-label[0,:,:,:,0].copy()))                
                train_dsc_tz.append(dice_3d(tz_pred,label[0,:,:,:,1].copy()))
                train_dsc_pz.append(dice_3d(pz_pred,label[0,:,:,:,2].copy()))

            # Validation Metrics
            for i in range(self.valid_ns):
                patient_case              = next(self.valid_gen)
                scan                      = np.expand_dims(patient_case[0]["image"],     axis=0)      
                label                     = np.expand_dims(patient_case[1]["detection"], axis=0)    
                all_pred                  = np.stack([np.array(detect_model.predict([scan]))[0,:,:,:,:] for _ in range(s)])               
                wg_pred, tz_pred, pz_pred = all_pred[:,:,:,:,0].copy(), all_pred[:,:,:,:,1].copy(), all_pred[:,:,:,:,2].copy()               

                # Variational Inference-Based Approximation of Predictive Distribution
                wg_pred, wg_var  = wg_pred.mean(axis=0), wg_pred.var(axis=0)
                tz_pred, tz_var  = tz_pred.mean(axis=0), tz_pred.var(axis=0)
                pz_pred, pz_var  = pz_pred.mean(axis=0), pz_pred.var(axis=0)

                # Append Predictions + Labels 
                valid_dsc_wg.append(dice_3d(1-wg_pred,1-label[0,:,:,:,0].copy()))                
                valid_dsc_tz.append(dice_3d(tz_pred,label[0,:,:,:,1].copy()))
                valid_dsc_pz.append(dice_3d(pz_pred,label[0,:,:,:,2].copy()))
                        
            print('-------------------------------------------------------------------------------------------------------------------------')
            print('Anatomy Segmentation - Training/Validation Performance    |||  ', datetime.datetime.now())
            print('-------------------------------------------------------------------------------------------------------------------------')
            print('Training Soft DSC [n='+str(self.train_ns)+']: '  +str(np.mean(train_dsc_wg))+' (WG); '\
                                                                    +str(np.mean(train_dsc_tz))+' (TZ); '\
                                                                    +str(np.mean(train_dsc_pz))+' (PZ)')
            print('Validation Soft DSC [n='+str(self.valid_ns)+']: '+str(np.mean(valid_dsc_wg))+' (WG); '\
                                                                    +str(np.mean(valid_dsc_tz))+' (TZ); '\
                                                                    +str(np.mean(valid_dsc_pz))+' (PZ)')
            print('-------------------------------------------------------------------------------------------------------------------------')
            
            # Update Lists
            epoch_points.append(self.epoch+1)                       
            all_train_dsc_wg.append(np.mean(train_dsc_wg))
            all_valid_dsc_wg.append(np.mean(valid_dsc_wg))  
            all_train_dsc_tz.append(np.mean(train_dsc_tz))
            all_valid_dsc_tz.append(np.mean(valid_dsc_tz))  
            all_train_dsc_pz.append(np.mean(train_dsc_pz))
            all_valid_dsc_pz.append(np.mean(valid_dsc_pz))  

            # Export Metrics
            metrics = pd.DataFrame(list(zip(epoch_points, all_train_dsc_wg,  all_valid_dsc_wg,\
                                                          all_train_dsc_tz,  all_valid_dsc_tz,\
                                                          all_train_dsc_pz,  all_valid_dsc_pz)), 
                                  columns=['epoch',      'train_dsc_wg',    'valid_dsc_wg',\
                                                         'train_dsc_tz',    'valid_dsc_tz',\
                                                         'train_dsc_pz',    'valid_dsc_pz'])
            while True: # To Counter {BlockingIOError: Resource temporarily unavailable}
                try:
                    metrics.to_excel(self.save_dir, encoding='utf-8', index=False)
                    break
                except: continue
        self.epoch += 1

