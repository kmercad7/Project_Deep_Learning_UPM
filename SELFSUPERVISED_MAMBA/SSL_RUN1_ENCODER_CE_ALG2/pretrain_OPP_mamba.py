from MambaSSLModel import MambaAutoencoderModel, HARMambaConfig, MambaDownstreamClassifier
from data.OPPORTUNITY_data import load_OPP_loco_data, data_split_OPP, make_loaders_OPP
from test_mamba import test_model, save_json 
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from mamba_ssm.modules.mamba2 import Mamba2
import mamba_ssm.modules.mamba2 as mm
from data.preprocessing import fit_labelencoder, Dataset_HAR
from datetime import datetime
import torch.nn.functional as F
import mamba_ssm
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import set_seed
import argparse
from torch.profiler import profile, ProfilerActivity
import time
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import random

print(f"Mamba version: {mamba_ssm.__version__}")
print(f"Path: {mm.__file__}")

parser = argparse.ArgumentParser(description = "supervised_mamba")
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--fold", type = int, required = True)
args = parser.parse_args()
if args.dataset == "OPP":
    if args.fold in [1, 2, 3, 4]:
        training_files, validation_files, test_files = data_split_OPP(args.fold)
    else:
        raise ValueError(f"Fold must be 1, 2, 3 or 4. Got {args.fold}")
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_OPP_loco_data(training_files, validation_files, test_files, verbose = True)

#  ----------------------------------------------------- SPAN MASKING ALGORITHM -----------------------------------------------------
def device_masking_algorithm(X_windows)-> [torch.Tensor, torch.Tensor] :
    '''
    Span one full IMU device with zero
    avoiding mismatch with 80/20 trick.
    '''
    B, Lw, Ft =  X_windows.shape
    #create empty mask
    mask = torch.zeros((B, Lw, Ft), dtype=torch.bool, device=X_windows.device)
    mask_zero = torch.zeros((B, Lw, Ft), dtype=torch.bool, device=X_windows.device)
    for b in range(B):
        I = set() 
        device_idx = random.randint(0, 4)
        cols = range(device_idx*9, device_idx*9+9) #select device
        I.update(cols) 
        
        # here add 80/20 to avoid mismatch.
        mask[b, :, list(I)] = True # need ALL the masked values for the loss.
        if np.random.rand() < 0.8:
            mask_zero[b, :, list(I)] = True
        
    X_masked = X_windows.clone()  
    X_masked[mask_zero] = 0

    return mask, X_masked    
#  ----------------------------------------------------- VALIDATION -----------------------------------------------------
@torch.no_grad()
def validate_model_PRETRAIN(model, val_loader, device, criterion):
    '''
    Validation: avg masked reconstruction loss.
    '''
    val_generator = torch.Generator()
    val_generator.manual_seed(42)
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    for x_batch, _ in val_loader:
        x_batch = x_batch.to(device, non_blocking = True)
        x_original = x_batch.clone()

        #mask, x_batch_masked = masking_span_algorithm(x_batch, 0.15, 5, generator = val_generator)        # masked inputs
        mask, x_batch_masked = device_masking_algorithm(x_batch)
        x_hat = model(x_batch_masked)                                                                     # forward Pass
        loss = criterion(x_hat[mask], x_original[mask])

        bsize = x_batch.size(0)
        total_loss += loss.item() * bsize
        total_samples += bsize
                
    return total_loss / total_samples
#  ---------------------------------------------------- LOSO TRAINING ---------------------------------------------------
for seed in [42, 58, 7, 128, 92]: # [42, 58, 7, 128, 92]
    g = set_seed(seed)
    train_loader, val_loader, test_loader, label_encoder = make_loaders_OPP(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator = g, verbose = True)
    
    #  ----------- TRAINING SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HARMambaConfig()
    model = MambaAutoencoderModel(config)
    model.to(device, non_blocking = True) 
    # ----------------------
    num_epochs = 50
    lr = 0.0006
    patience = 5
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
    
    #  ----------- TRAINING -----------
    model_name = f"SSL_models_pt/SSL_model_lr6e4_OPP_fold{args.fold}_seed{seed}.pt"
    epoch_history = []
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    bad_epochs = 0
    prof_out = None
    with open(f"logs/SSL_training_lr6e4_OPP_fold{args.fold}.txt", "a") as log_file:
        log_file.write(f"\nTRAINING STARTING AT: {datetime.now()}\n")
        log_file.write(f"Model: {model_name} | SEED: {seed}\n")
        log_file.flush()
        for epoch in range(num_epochs):
            epoch_start = time.time() # START EPOCH TIME
            model.train()
            total_loss = 0.0
            total_samples = 0
    
            loop = tqdm(train_loader, desc= f"Epoch {epoch+1}/{num_epochs}")
            for _, (x_batch, _) in enumerate(loop):
                x_batch = x_batch.to(device, non_blocking = True)   
                
                optimizer.zero_grad()                                             # clear previous gradients
                x_original = x_batch.clone()
                #mask, x_batch_masked = masking_span_algorithm(x_batch, 0.30, 5)
                mask, x_batch_masked = device_masking_algorithm(x_batch)
                x_hat = model(x_batch_masked)
                loss = criterion(x_hat[mask], x_original[mask])                  
                loss.backward()                                                   # backward pass: compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
                optimizer.step()                                                  # update model weights

                bsize = x_batch.size(0)
                total_loss += loss.item() * bsize                                 # Total loss contribution of this batch
                total_samples += bsize
                loop.set_postfix(loss= f"{total_loss/total_samples:.4f}")
    
            train_loss = total_loss / total_samples
            val_loss = validate_model_PRETRAIN(model, val_loader, device, criterion)

            epoch_time = time.time() - epoch_start # END EPOCH TIME
            epoch_history.append({
                "epoch": epoch + 1,
                "tr_loss": float(train_loss),
                "val_loss": float(val_loss)              
            })
            print(f"\nEpoch: {epoch+1}/{num_epochs} | tr_Loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | epoch_time: {epoch_time:.2f}s")  
            log_file.write(f"Epoch: {epoch+1}/{num_epochs} | tr_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | epoch_time: {epoch_time:.2f}s\n")
            log_file.flush()
                
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = float(val_loss)
                best_epoch = epoch + 1                   
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best Validation Loss: {best_val_loss:.4f}")
                    break
                  
        if best_state is not None:
            torch.save(best_state, model_name)      
        log_file.write(f"TRAINING ENDING AT: {datetime.now()}\n")                               
#  ----------------------------------------------------------------------------------------------------------------------