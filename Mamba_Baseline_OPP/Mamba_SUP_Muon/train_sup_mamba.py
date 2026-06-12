from MambaClassificationModel import MambaClassificationModel, HARMambaConfig
from data.OPPORTUNITY_data import load_OPP_loco_data, data_split_OPP, make_loaders_OPP
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data.preprocessing import fit_labelencoder
from test_mamba import test_model, save_json 
from mamba_ssm.modules.mamba2 import Mamba2
import mamba_ssm.modules.mamba2 as mm
from datetime import datetime
import torch.nn.functional as F
import mamba_ssm
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import set_seed
import argparse
import json
import os
from torch.profiler import profile, ProfilerActivity
import time 
from muon import Muon, SingleDeviceMuonWithAuxAdam
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

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

X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_OPP_loco_data(training_files, validation_files, test_files, fold_id = args.fold, verbose = True)

#  ----------------------------------------------------- VALIDATION -----------------------------------------------------
@torch.no_grad()
def validate_model(model, val_loader, device, criterion):
    '''
    Validation: avg loss per window and accuracy per window.
    '''
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)
        logits_ = model(x_batch)                                               # forward Pass
        loss = criterion(logits_, y_batch)                                     # computes mean batch loss

        bsize = y_batch.size(0)
        total_loss += loss.item() * bsize                                      # Total loss contribution of this batch

        predictions = logits_.argmax(dim = -1)                                 # gets the predicted class for each window
        total_correct += (predictions == y_batch).sum().item()                 # counts correct predictions
        total_samples += bsize

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names = ["STAND", "WALK", "SIT", "LIE"])
    return total_loss / total_samples, total_correct / total_samples, report
#  ----------------------------------------------------------------------------------------------------------------------
#  ---------------------------------------------------- LOSO TRAINING ---------------------------------------------------
muon_params = []
adamw_params = []
for seed in [42, 58, 7, 128, 92]: # [42, 58, 7, 128, 92]
    g = set_seed(seed)
    train_loader, val_loader, test_loader, label_encoder = make_loaders_OPP(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator = g, verbose = True)
    
    #  ----------- TRAINING SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(len(label_encoder.classes_))
    config = HARMambaConfig()
    model = MambaClassificationModel(config, num_classes = num_classes)
    model.to(device, non_blocking = True) 
    # -- Reset Peak Stats --
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats() 
    # ----------------------
    num_epochs = 30
    lr = 0.0006
    patience = 5
    #WCE
    #label_encoder = fit_labelencoder(X_windows, y_windows)
    #y_train_encoded = label_encoder.transform(np.asarray(y_windows))
    #counts = np.bincount(y_train_encoded, minlength = num_classes)
    #weights = torch.tensor(1.0 / counts, dtype=torch.float32, device = device)
    #criterion = nn.CrossEntropyLoss(weight = weights)
    criterion = nn.CrossEntropyLoss()

    #  ----------- MUON -----------
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
    
        use_muon = (
            param.ndim == 2 and (
                name == "backbone.input_proj.weight"
                or name.endswith(".mixer.in_proj.weight")
                or name.endswith(".mixer.out_proj.weight")
                or name.endswith(".mlp.fc1.weight")
                or name.endswith(".mlp.fc2.weight"))
        )
    
        if use_muon:
            muon_params.append(param)
            print("Muon :", name, tuple(param.shape))
        else:
            adamw_params.append(param)
            print("AdamW:", name, tuple(param.shape))    

    optimizer = SingleDeviceMuonWithAuxAdam([
        dict(params=muon_params, use_muon=True, lr=0.01, momentum=0.95, weight_decay=0.0),
        dict(params=adamw_params,use_muon=False, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,)
    ])
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
    #  ----------- TRAINING -----------
    model_name = f"models_pt/model_OPP_fold{args.fold}_seed{seed}.pt"
    epoch_history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = None
    best_state = None
    bad_epochs = 0
    prof_out = None
    with open(f"logs/training_OPP_fold{args.fold}.txt", "a") as log_file:
        log_file.write(f"\nTRAINING STARTING AT: {datetime.now()}\n")
        log_file.write(f"Model: {model_name} | SEED: {seed}\n")
        log_file.flush()
        for epoch in range(num_epochs):
            epoch_start = time.time() # START EPOCH TIME
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
    
            loop = tqdm(train_loader, desc= f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (x_batch, y_batch) in enumerate(loop):
                x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)    
                #################### PROFILING ####################
                if device.type == "cuda" and epoch == 0 and batch_idx == 1:
                    with profile(
                        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes = True,
                        profile_memory = True
                    ) as prof_:
                        optimizer.zero_grad()                                             # clear previous gradients
                        logits_ = model(x_batch)                                          # forward pass
                        loss = criterion(logits_, y_batch)                                # computes mean batch loss
                        loss.backward()                                                   # backward pass: compute gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
                        optimizer.step()
                        torch.cuda.synchronize()
                    prof_out = prof_.key_averages().table(sort_by = "self_cuda_time_total", row_limit = 10)
                else:
                    optimizer.zero_grad()                                             # clear previous gradients
                    logits_ = model(x_batch)                                          # forward pass
                    loss = criterion(logits_, y_batch)                                # computes mean batch loss
                    loss.backward()                                                   # backward pass: compute gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
                    optimizer.step()                                                  # update model weights
                ###################################################    
                bsize = y_batch.size(0)
                total_loss += loss.item() * bsize                                 # Total loss contribution of this batch
                
                predictions = logits_.argmax(dim = -1)                            # gets the predicted class for each window: picks the highest score in the last dimension (one highest score per window among the 4 classes)
                total_correct += (predictions == y_batch).sum().item()            # number of correct predictions
                total_samples += bsize
                loop.set_postfix(loss= f"{total_loss/total_samples:.4f}", acc=f"{total_correct/total_samples:.4f}")
    
            train_loss, train_acc = total_loss / total_samples, total_correct / total_samples
            val_loss, val_acc, report = validate_model(model, val_loader, device, criterion)
            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_time = time.time() - epoch_start # END EPOCH TIME
            epoch_history.append({
                "epoch": epoch + 1,
                "tr_loss": float(train_loss),
                "tr_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc)                
            })
            print(f"\nEpoch: {epoch+1}/{num_epochs} | tr_Loss: {train_loss:.4f} | tr_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | epoch_time: {epoch_time:.2f}s")
            if epoch == 0 and prof_out is not None:
                print("--- CUDA PROFILE --- (epoch 1, batch 2)")
                print(prof_out)    
            log_file.write(f"Epoch: {epoch+1}/{num_epochs} | tr_loss: {train_loss:.4f} | tr_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | epoch_time: {epoch_time:.2f}s\n")
            log_file.flush()
                
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = float(val_loss)
                best_val_acc = float(val_acc)
                best_epoch = epoch + 1                   
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best Validation Loss: {best_val_loss:.4f}")
                    break
                    
        # Load Best Model !            
        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, model_name)
            # Run val on best model to generate report
            _, _, report = validate_model(model, val_loader, device, criterion) 
            print(report)
            log_file.write(f"\nBest Model Validation Report: \n{str(report)}")        
        log_file.write(f"TRAINING ENDING AT: {datetime.now()}\n")                           
        
        #  ----------- TEST -----------  
        # -- Reset Peak Stats --
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak GPU memory allocatated: {peak_mem_gb:.2f}GB\n")
        # ----------------------        
        acc, report, f1, conf_matrix = test_model(model, test_loader, device)
        seed_result = {
            "seed": int(seed),
            "fold": int(args.fold),
            "history": epoch_history,
            "summary": {
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "test_accuracy": float(acc),
                "test_report": report,
                "test_f1": float(f1),
                "test_conf_matrix": conf_matrix.tolist()               
            }
        }
        save_json(args.dataset, args.fold, seed, seed_result)
        print(f"{'-'*90}")
        print(f"Test Results:\n Accuracy: {acc}\n Report:\n {report}\n F1: {f1}\n Confusion Matrix:\n {conf_matrix}")
        log_file.write(f"\nTest Results:\n Accuracy: {acc}\n Report:\n {report}\n F1: {f1}\n Confusion Matrix:\n {conf_matrix}")
#  ----------------------------------------------------------------------------------------------------------------------