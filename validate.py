# %%
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import SimpleITK as sitk
import os
import numpy as np
import scipy.ndimage
import time
import os
from os.path import abspath, join, isfile
import cv2
import skimage
from skimage.measure import regionprops
from sklearn.metrics import log_loss
from shutil import copyfile
from scipy.stats import entropy
import tensorflow as tf
# import model.voxelmorph.tf as vxm
import model.unets as unets
from data_generators2 import custom_data_generator
from callbacks import dice_3d
from misc import setup_device
import warnings
import multiprocessing
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils import pred_to_label
from medpy.metric.binary import hd, asd, ravd
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

# Extract Largest Connected Component
def getLargest(segmentation, structure=np.ones((5,5,5))):
    bin_seg = np.ceil(np.array(segmentation).copy()).astype(np.uint8)
    try:
        labels    = skimage.measure.label(bin_seg)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return (largestCC.astype(np.float32)*segmentation).astype(np.float32)
    except: return segmentation.astype(np.float32)



parser = argparse.ArgumentParser()

parser.add_argument("--FOLDS", required=False, type=str, help="Fold numbers")
parser.add_argument("--NAME",  required=False, default='EXP_NAME', type=str, help="?Experiment")

args    = parser.parse_args()
FOLD    = args.FOLDS
EXP     = args.NAME 
print(vars(args))



CODE_BASE      = abspath('/mnt/zonal_segmentation')
probabilistic = True
      
DATA_ProstateX_VAL   = { 'f0': join(CODE_BASE, f'data_feed/ProstateX_valid-fold-0.xlsx'),
                         'f1': join(CODE_BASE, f'data_feed/ProstateX_valid-fold-1.xlsx'),
                         'f2': join(CODE_BASE, f'data_feed/ProstateX_valid-fold-2.xlsx'),
                         'f3': join(CODE_BASE, f'data_feed/ProstateX_valid-fold-3.xlsx'),
                         'f4': join(CODE_BASE, f'data_feed/ProstateX_valid-fold-4.xlsx'),
                         'all': join(CODE_BASE, f'data_feed/ProstateX.xlsx')}

DATA_NTNU_VAL        = { 'f0': join(CODE_BASE, f'data_feed/External_NTNU.xlsx'),
                         'f1': join(CODE_BASE, f'data_feed/External_NTNU.xlsx'),
                         'f2': join(CODE_BASE, f'data_feed/External_NTNU.xlsx'),
                         'f3': join(CODE_BASE, f'data_feed/External_NTNU.xlsx'),
                         'f4': join(CODE_BASE, f'data_feed/External_NTNU.xlsx'),
                         'all': join(CODE_BASE, f'data_feed/External_NTNU.xlsx')}

if EXP=='ZPX_P':
        DATA           = DATA_NTNU
        OUTPUT_FILE    = 'uncertainity_metrics_results.csv'
        WITH_GT        = True
        OUTPUT_DIR     = join(CODE_BASE, 'experiments', EXP) 
        MODEL_WEIGHTS  = {'f0': join(CODE_BASE, 'experiments', EXP, 'f0', 'best_model.h5'),
                          'f1': join(CODE_BASE, 'experiments', EXP, 'f1', 'best_model.h5'),
                          'f2': join(CODE_BASE, 'experiments', EXP, 'f2', 'best_model.h5'),
                          'f3': join(CODE_BASE, 'experiments', EXP, 'f3', 'best_model.h5'),
                          'f4': join(CODE_BASE, 'experiments', EXP, 'f4', 'best_model.h5') }


SAVE_PREDICTION_NPY = True
PROBA_ITER     = 20


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Initialize Metric Lists
all_dsc_wg,  all_dsc_tz,  all_dsc_pz  = [],[],[]
all_nll_wg,  all_nll_tz,  all_nll_pz  = [],[],[]
all_ent_wg,  all_ent_tz,  all_ent_pz  = [],[],[]
fold_dsc_wg, fold_dsc_tz, fold_dsc_pz = [],[],[]
fold_nll_wg, fold_nll_tz, fold_nll_pz = [],[],[]
fold_ent_wg, fold_ent_tz, fold_ent_pz = [],[],[]

# Validation Metrics Per Fold
if isinstance(FOLD, str): FOLD = [FOLD]

for fld in FOLD:
    print('VALIDATING:', fld, DATA[fld])
    print('MODEL:', MODEL_WEIGHTS[fld])

    if not os.path.exists(join(OUTPUT_DIR, fld)):
        os.mkdir(join(OUTPUT_DIR, fld))
    if SAVE_PREDICTION_NPY:
        NPY_PATH  = join(CODE_BASE, f'data/{EXP}/{fld}')
        os.makedirs(NPY_PATH, exist_ok=True)


    VALID_XLSX   = DATA[fld]
    detect_model = unets.networks.M1.load(path=MODEL_WEIGHTS[fld]).get_detect_model()
    valid_gen    = custom_data_generator(data_xlsx=VALID_XLSX, train_obj='zonal', probabilistic=probabilistic, test=not WITH_GT)

    asd_wg, asd_tz, asd_pz                         = [],[],[]
    hd_wg, hd_tz, hd_pz                            = [],[],[]
    ravd_wg, ravd_tz, ravd_pz                      = [],[],[]
    dsc_wg, dsc_tz, dsc_pz                         = [],[],[]
    dsc_wg_base, dsc_tz_base, dsc_pz_base          = [],[],[]
    dsc_wg_mid,  dsc_tz_mid,  dsc_pz_mid           = [],[],[]
    dsc_wg_apex, dsc_tz_apex, dsc_pz_apex          = [],[],[]

    mean_nll_wg, mean_nll_tz, mean_nll_pz          = [],[],[]
    mean_ent_wg, mean_ent_tz, mean_ent_pz          = [],[],[]
    mean_ent_in_wg, mean_ent_in_tz, mean_ent_in_pz = [],[],[]
    vol_pr_wg, vol_pr_tz, vol_pr_pz                = [],[],[]
    vol_gt_wg, vol_gt_tz, vol_gt_pz                = [],[],[]
    patient_ids                                    = []
   

    # Validation Metrics Per Scan
    all_patient_ids = pd.read_excel(VALID_XLSX)['s-id'].values
    for i in tqdm(range(len(all_patient_ids))):
        patient_case              = next(valid_gen)
        scan                      = np.expand_dims(patient_case[0]["image"],     axis=0)      
        if WITH_GT: label         = np.expand_dims(patient_case[1]["detection"], axis=0)    
        all_pred                  = np.stack([np.array(detect_model.predict([scan]))[0,:,:,:,:] for _ in range(PROBA_ITER)])               
        wg_pred, tz_pred, pz_pred = all_pred[:,:,:,:,0].copy(), all_pred[:,:,:,:,1].copy(), all_pred[:,:,:,:,2].copy()               

        # Save precitions as numpy   
        if SAVE_PREDICTION_NPY:
            generated_label = pred_to_label(all_pred, binary=False)
            np.save(join(NPY_PATH, str(all_patient_ids[i])+'.npy'), generated_label)


        patient_ids.append(all_patient_ids[i])
        # Variational Inference-Based Approximation of Predictive Distribution
        wg_pred, wg_var = wg_pred.mean(axis=0), wg_pred.var(axis=0)
        tz_pred, tz_var = tz_pred.mean(axis=0), tz_pred.var(axis=0)
        pz_pred, pz_var = pz_pred.mean(axis=0), pz_pred.var(axis=0)

        # Binarized Predictions
        hard_mask = np.argmax(all_pred.mean(axis=0), axis=-1)
        wg_bin, tz_bin, pz_bin = np.zeros_like(hard_mask), np.zeros_like(hard_mask), np.zeros_like(hard_mask)
        wg_bin[hard_mask>0], tz_bin[hard_mask==1], pz_bin[hard_mask==2] = 1,1,1

        # Append Segment-Level Soft DSC of WG/TZ/PZ Segmentation 

        wg_bin_largestcc = getLargest(wg_bin)
        tz_bin_largestcc = getLargest(tz_bin)
        pz_bin_largestcc = getLargest(pz_bin)

        vol_pr_wg.append((np.count_nonzero(wg_bin_largestcc))*0.5*0.5*3.6/1000)
        vol_pr_tz.append((np.count_nonzero(tz_bin_largestcc))*0.5*0.5*3.6/1000)
        vol_pr_pz.append((np.count_nonzero(pz_bin_largestcc))*0.5*0.5*3.6/1000)

        if WITH_GT:
            # Average Volume
            vol_gt_wg.append((np.count_nonzero(1-label[0,:,:,:,0]))*0.5*0.5*3.6/1000)
            vol_gt_tz.append((np.count_nonzero(  label[0,:,:,:,1]))*0.5*0.5*3.6/1000)
            vol_gt_pz.append((np.count_nonzero(  label[0,:,:,:,2]))*0.5*0.5*3.6/1000)

            # Average DSC
            dsc_wg.append(dice_3d(wg_bin_largestcc, 1-label[0,:,:,:,0]))                
            dsc_tz.append(dice_3d(tz_bin_largestcc,   label[0,:,:,:,1]))
            dsc_pz.append(dice_3d(pz_bin_largestcc,   label[0,:,:,:,2]))

            hd_wg.append(hd(wg_bin_largestcc, 1-label[0,:,:,:,0], voxelspacing=(3.6,0.5,0.5)))                
            hd_tz.append(hd(tz_bin_largestcc,   label[0,:,:,:,1], voxelspacing=(3.6,0.5,0.5)))
            hd_pz.append(hd(pz_bin_largestcc,   label[0,:,:,:,2], voxelspacing=(3.6,0.5,0.5)))

            asd_wg.append(asd(wg_bin_largestcc, 1-label[0,:,:,:,0], voxelspacing=(3.6,0.5,0.5)))                
            asd_tz.append(asd(tz_bin_largestcc,   label[0,:,:,:,1], voxelspacing=(3.6,0.5,0.5)))
            asd_pz.append(asd(pz_bin_largestcc,   label[0,:,:,:,2], voxelspacing=(3.6,0.5,0.5)))

            ravd_wg.append(ravd(wg_bin_largestcc, 1-label[0,:,:,:,0]))                
            ravd_tz.append(ravd(tz_bin_largestcc,   label[0,:,:,:,1]))
            ravd_pz.append(ravd(pz_bin_largestcc,   label[0,:,:,:,2]))

            # WG slide numbers
            wg_indexes = list() # indexes that have WG class
            for n in range(0, label.shape[1]):
                if not np.all(label[0,n,:,:,0]):
                    wg_indexes.append(n)
            
            wg_indexes = np.array(wg_indexes)
            p34_ind = int(np.floor(np.percentile(wg_indexes, 34)))
            p67_ind = int(np.floor(np.percentile(wg_indexes, 67)))

            # Average DSC in Base, Mid and Apex
            dsc_wg_base.append(dice_3d(wg_bin_largestcc[0:p34_ind+1,:,:], 1-label[0,0:p34_ind+1,:,:,0]))                
            dsc_tz_base.append(dice_3d(tz_bin_largestcc[0:p34_ind+1,:,:],   label[0,0:p34_ind+1,:,:,1]))
            dsc_pz_base.append(dice_3d(pz_bin_largestcc[0:p34_ind+1,:,:],   label[0,0:p34_ind+1,:,:,2]))

            dsc_wg_mid.append(dice_3d(wg_bin_largestcc[p34_ind+1:p67_ind+1,:,:], 1-label[0,p34_ind+1:p67_ind+1,:,:,0]))                
            dsc_tz_mid.append(dice_3d(tz_bin_largestcc[p34_ind+1:p67_ind+1,:,:],   label[0,p34_ind+1:p67_ind+1,:,:,1]))
            dsc_pz_mid.append(dice_3d(pz_bin_largestcc[p34_ind+1:p67_ind+1,:,:],   label[0,p34_ind+1:p67_ind+1,:,:,2]))

            dsc_wg_apex.append(dice_3d(wg_bin_largestcc[p67_ind+1:,:,:], 1-label[0,p67_ind+1:,:,:,0]))                
            dsc_tz_apex.append(dice_3d(tz_bin_largestcc[p67_ind+1:,:,:],   label[0,p67_ind+1:,:,:,1]))
            dsc_pz_apex.append(dice_3d(pz_bin_largestcc[p67_ind+1:,:,:],   label[0,p67_ind+1:,:,:,2]))

            # Append Segment-Level Negative Log-Likelighood of WG/TZ/PZ Segmentation    
            mean_nll_wg.append(log_loss((1-label[0,:,:,:,0]).flatten().astype(np.float64), ((1-wg_pred)*getLargest(np.ceil((1-wg_pred)))).flatten().astype(np.float64)))                
            mean_nll_tz.append(log_loss((label[0,:,:,:,1]).flatten().astype(np.float64),   ((tz_pred)  *getLargest(np.ceil((tz_pred)))).flatten().astype(np.float64)))
            mean_nll_pz.append(log_loss((label[0,:,:,:,2]).flatten().astype(np.float64),   ((pz_pred)  *getLargest(np.ceil((pz_pred)))).flatten().astype(np.float64)))

        # Append Segment-Level Predictive Entropy of WG/TZ/PZ Segmentation 
        ent_wg, ent_tz, ent_pz = np.zeros_like(wg_pred), np.zeros_like(wg_pred), np.zeros_like(wg_pred)

        for e in range(wg_pred.shape[0]):
            ent_wg[e] = entropy(np.stack((wg_pred[e],  1-wg_pred[e])), axis=0)
            ent_tz[e] = entropy(np.stack((1-tz_pred[e],  tz_pred[e])), axis=0)             
            ent_pz[e] = entropy(np.stack((1-pz_pred[e],  pz_pred[e])), axis=0) 

        # Average Entropy
        mean_ent_wg.append(np.mean(ent_wg))
        mean_ent_tz.append(np.mean(ent_tz))
        mean_ent_pz.append(np.mean(ent_pz))

        # Average Entropy Inside Predicted Segment
        wg_largestcc_ceil = getLargest(np.ceil(1-wg_pred))
        tz_largestcc_ceil = getLargest(np.ceil(tz_pred))
        pz_largestcc_ceil = getLargest(np.ceil(pz_pred))

        mean_ent_in_wg.append(np.sum(ent_wg*wg_largestcc_ceil)   / np.sum(wg_largestcc_ceil  ))
        mean_ent_in_tz.append(np.sum(ent_tz*tz_largestcc_ceil)   / np.sum(tz_largestcc_ceil  ))
        mean_ent_in_pz.append(np.sum(ent_pz*pz_largestcc_ceil)   / np.sum(pz_largestcc_ceil  ))


        if (i+1)%50==0 or (i+1)==len(all_patient_ids):
            df_columns_names = ({'s-id':       patient_ids,
                                'vol_WG_pr':   vol_pr_wg,  
                                'vol_TZ_pr':   vol_pr_tz,  
                                'vol_PZ_pr':   vol_pr_pz,
                                'ent_mean_WG': mean_ent_wg, 
                                'ent_mean_TZ': mean_ent_tz, 
                                'ent_mean_PZ': mean_ent_pz,
                                'ent_in_WG':   mean_ent_in_wg, 
                                'ent_in_TZ':   mean_ent_in_tz, 
                                'ent_in_PZ':   mean_ent_in_pz})

                                
            if WITH_GT:
                df_columns_names.update({'WG_dsc': dsc_wg, 
                                        'TZ_dsc':      dsc_tz, 
                                        'PZ_dsc':      dsc_pz,
                                        'WG_base_dsc': dsc_wg_base, 
                                        'TZ_base_dsc': dsc_tz_base, 
                                        'PZ_base_dsc': dsc_pz_base,  
                                        'WG_mid_dsc':  dsc_wg_mid, 
                                        'TZ_mid_dsc':  dsc_tz_mid, 
                                        'PZ_mid_dsc':  dsc_pz_mid, 
                                        'WG_apex_dsc': dsc_wg_apex, 
                                        'TZ_apex_dsc': dsc_tz_apex, 
                                        'PZ_apex_dsc': dsc_pz_apex,                       
                                        'vol_WG_gt':   vol_gt_wg,    
                                        'vol_TZ_gt':   vol_gt_tz,    
                                        'vol_PZ_gt':   vol_gt_pz,
                                        'nll_wg':      mean_nll_wg,
                                        'nll_tz':      mean_nll_tz,
                                        'nll_pz':      mean_nll_pz,
                                        'WG_hd':       hd_wg, 
                                        'TZ_hd':       hd_tz, 
                                        'PZ_hd':       hd_pz,
                                        'WG_asd':      asd_wg, 
                                        'TZ_asd':      asd_tz, 
                                        'PZ_asd':      asd_pz,
                                        'WG_ravd':     ravd_wg, 
                                        'TZ_ravd':     ravd_tz, 
                                        'PZ_ravd':     ravd_pz,})

            df = pd.DataFrame(df_columns_names)
            df.to_csv(join(OUTPUT_DIR, fld, OUTPUT_FILE))
            print('Saved metrics to:', join(OUTPUT_DIR, fld, OUTPUT_FILE))

 