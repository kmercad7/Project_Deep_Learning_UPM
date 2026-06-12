from .data_pipeline import (load_pamap2, remove_zero_label_rows, incomplete_labeled_rows, remove_all_nan_rows, sliding_window, divide_features_labels, acc_data_scaling_pam, mag_data_norm_pam, mag_data_rotation_pam, downsample_pam, filter_loco_pam, interpolation_pam)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .preprocessing import fit_labelencoder, Dataset_HAR
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
import torch

def data_split_PAM(fold_id: int = 1) -> tuple[list, list, list]:
    '''
    Predefined subject split for PAMAP2 dataset.
    For a given fold_id, the corresponding subjects are held out as test.
    
    fold_id 1 → test: 101, 102 | val: 103, 104 | train: 105, 106, 107, 108
    fold_id 2 → test: 107, 108 | val: 101, 102 | train: 103, 104, 105, 106
    fold_id 3 → test: 105, 106 | val: 107, 108 | train: 101, 102, 103, 104
    fold_id 4 → test: 103, 104 | val: 105, 106 | train: 107, 108, 101, 102
    '''
    splits_pam = {
        1: {"test": ["101", "102"], "val": ["103", "104"], "train": ["105", "106", "107", "108"]},
        2: {"test": ["107", "108"], "val": ["101", "102"], "train": ["103", "104", "105", "106"]},
        3: {"test": ["105", "106"], "val": ["107", "108"], "train": ["101", "102", "103", "104"]},
        4: {"test": ["103", "104"], "val": ["105", "106"], "train": ["107", "108", "101", "102"]}
    }

    split = splits_pam[fold_id]
    base_path = '../DATASETS/PAMAP2_Dataset/Protocol/'

    training_files = [base_path + 'subject' + subject + '.dat' for subject in split["train"]]
    validation_files = [base_path + 'subject' + subject + '.dat' for subject in split["val"]]
    test_files = [base_path + 'subject' + subject + '.dat' for subject in split["test"]]

    return training_files, validation_files, test_files

def print_post_standardize_stats(df: pd.DataFrame, name: str, acc_gyro_cols: list[int]):
    x = df.iloc[:, acc_gyro_cols].to_numpy(dtype=np.float64)
    ch_mean = x.mean(axis=0)
    ch_std = x.std(axis=0)

    print(f"\n{'-'*30} {name} ACC/GYRO after StandardScaler {'-'*30}")
    print(f"global mean: {x.mean():.6f}")
    print(f"global std : {x.std():.6f}")
    print(f"max |channel mean| : {np.abs(ch_mean).max():.6f}")
    print(f"median |channel mean| : {np.median(np.abs(ch_mean)):.6f}")
    print(f"channel std range : [{ch_std.min():.6f}, {ch_std.max():.6f}]")

def print_mag_rotation_stats(Xw: np.ndarray, name: str):
    mag_cols = [axis + offset for offset in range(6, 27, 9) for axis in range(3)]
    M = Xw[:, :, mag_cols]                     # [W, L, 15]
    per_window_mean = M.mean(axis=1)           # [W, 15]
    flat = M.reshape(-1, M.shape[-1])          # [W*L, 15]

    print(f"\n{'-'*30} {name} MAG after window demeaning {'-'*30}")
    print(f"max |window mean|  : {np.abs(per_window_mean).max():.6e}")
    print(f"mean |window mean| : {np.abs(per_window_mean).mean():.6e}")
    print(f"channel global mean range : [{flat.mean(axis=0).min():.6e}, {flat.mean(axis=0).max():.6e}]")
    print(f"channel std range         : [{flat.std(axis=0).min():.6f}, {flat.std(axis=0).max():.6f}]")

def print_window_label_distribution(yw: np.ndarray, name: str):
    vals, cnts = np.unique(yw, return_counts=True)
    total = len(yw)
    print(f"\n{name} window label distribution")
    for v, c in zip(vals, cnts):
        print(f"label {v}: {c} ({c/total:.4f})")

def print_top_shifted_channels(df, cols, name, top_k=10):
    X = df.iloc[:, cols].to_numpy(dtype=np.float64)
    ch_mean = X.mean(axis=0)
    ch_std = X.std(axis=0)

    print(f"\n{name} top channels by |mean|:")
    for idx in np.argsort(np.abs(ch_mean))[::-1][:top_k]:
        print(f"channel {cols[idx]} | mean={ch_mean[idx]:.6f} | std={ch_std[idx]:.6f}")

    print(f"\n{name} top channels by std deviation from 1:")
    for idx in np.argsort(np.abs(ch_std - 1))[::-1][:top_k]:
        print(f"channel {cols[idx]} | mean={ch_mean[idx]:.6f} | std={ch_std[idx]:.6f}")

def print_outlier_rates(df, cols, name, thr=5.0):
    X = df.iloc[:, cols].to_numpy(dtype=np.float64)
    rates = (np.abs(X) > thr).mean(axis=0)
    print(f"\n{name} outlier rates |z|>{thr}:")
    for idx in np.argsort(rates)[::-1][:10]:
        print(f"channel {cols[idx]} | rate={rates[idx]:.6f}")
        
def load_PAM_loco_data(training_files, validation_files, test_files, verbose = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load PAMAP2 dataset and process all the data for the model
    '''
    training_data = load_pamap2(training_files, add_group_id = True)
    validation_data = load_pamap2(validation_files, add_group_id = True)
    test_data = load_pamap2(test_files, add_group_id = True)
    
    # ----- Selection of the Columns -----
    training_data_selected = filter_loco_pam(training_data)
    validation_data_selected = filter_loco_pam(validation_data)
    test_data_selected = filter_loco_pam(test_data)

    # ----- Clean label 0 ------ #MOD: REMOVE ALL ROWS WITH LABEL 0. 
    training_data_nan = remove_zero_label_rows(training_data_selected)
    validation_data_nan = remove_zero_label_rows(validation_data_selected)
    test_data_nan = remove_zero_label_rows(test_data_selected)
    if verbose:
        print(f"{'-'*90}")
        print(f"Sensor subset training shape: {training_data_nan.shape}\nSensor Validation training shape: {validation_data_nan.shape}\nSensor Test training shape: {test_data_nan.shape}")  
    
     # ----- For verification if there are rows with any NaN value  ----- 
    training_nan = incomplete_labeled_rows(training_data_nan)
    validation_nan = incomplete_labeled_rows(validation_data_nan)
    test_nan = incomplete_labeled_rows(test_data_nan)
    if verbose:
        print(f"{'-'*70}")
        print(f"Rows containing NaNs - training: {training_nan}\nRows containing NaNs - validation: {validation_nan}\nRows containing NaNs - test: {test_nan}")  
        
    # ----- Reset index ----- 
    training_data_nan = training_data_nan.reset_index(drop=True)
    validation_data_nan = validation_data_nan.reset_index(drop=True)
    test_data_nan = test_data_nan.reset_index(drop=True) 
    if verbose:
        print(f"{'-'*90}")
        print(f"Reindexed training shape: {training_data_nan.shape}\nReindexed validation shape: {validation_data_nan.shape}\nReindexed test shape: {test_data_nan.shape}")
   
    # ----- Interpolation -----
    training_data_cleaned = interpolation_pam(training_data_nan, max_gap=30)
    validation_data_cleaned = interpolation_pam(validation_data_nan, max_gap=30)
    test_data_cleaned = interpolation_pam(test_data_nan, max_gap=30)      
    if verbose:
        print(f"{'-'*90}")
        print(f"After Interpolation. Training shape: {training_data_cleaned.shape}\nAfter Interpolation. Validation shape: {validation_data_cleaned.shape}\nAfter Interpolation. Test shape: {test_data_cleaned.shape}")   
    
# --------------------------------------------------
# 1) physics-based scaling
# --------------------------------------------------
    if verbose: 
        print("\n")
        print("Descriptive statistics for ACC channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 0:3].describe())
        print("\nDescriptive statistics for GYRO channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 3:6].describe())
        print("\nDescriptive statistics for MAG channels (DEVICE_1) - training set")
        print(training_data_cleaned.iloc[:, 6:9].describe())
        print(f"{'-'*90}")
    
    # SCALING ACC
    training_data_scaled = acc_data_scaling_pam(training_data_cleaned)
    validation_data_scaled = acc_data_scaling_pam(validation_data_cleaned)
    test_data_scaled = acc_data_scaling_pam(test_data_cleaned)
    
    # NORM MAG               
    training_data_scaled = mag_data_norm_pam(training_data_scaled)
    validation_data_scaled = mag_data_norm_pam(validation_data_scaled)
    test_data_scaled = mag_data_norm_pam(test_data_scaled)
    
    if verbose:
        print("\n")
        print("Descriptive statistics for ACC channels (DEVICE_1) - training set")
        print(training_data_scaled.iloc[:, 0:3].describe())
        print("\nDescriptive statistics for scaled GYRO channels (DEVICE_1) - training set")
        print(training_data_scaled.iloc[:, 3:6].describe())
        print("\nDescriptive statistics for MAG channels (DEVICE_1) - training set")
        print(training_data_scaled.iloc[:, 6:9].describe())

# --------------------------------------------------
# 2) columns to standardize ACC AND GYRO
# --------------------------------------------------    
    scaler = StandardScaler()
    acc_gyro_cols = [axis + offset for offset in range(0, 27, 9) for axis in range(6)]
    
    training_data_scaled.iloc[:, acc_gyro_cols] = scaler.fit_transform(training_data_scaled.iloc[:, acc_gyro_cols].values)
    validation_data_scaled.iloc[:, acc_gyro_cols] = scaler.transform(validation_data_scaled.iloc[:, acc_gyro_cols].values)
    test_data_scaled.iloc[:, acc_gyro_cols] = scaler.transform(test_data_scaled.iloc[:, acc_gyro_cols].values)
    ############################################
    # Stratify
    combined_eval = pd.concat([validation_data_scaled, test_data_scaled], axis = 0, ignore_index = True) 
    y = combined_eval.iloc[:, -2].values       # Labels
    groups = combined_eval["group_id"].values  # File id Name
        
    sgkf = StratifiedGroupKFold(n_splits = 2, shuffle = True, random_state = 42)
    val_idx, test_idx = next(sgkf.split(combined_eval, y, groups))
    print(f"{'-'*20}STRATIY VAL/TEST{'-'*20}")
    print("Validation GROUPS")
    print(combined_eval.iloc[val_idx]["group_id"].value_counts().sort_index())
    print("Test GROUPS")
    print(combined_eval.iloc[test_idx]["group_id"].value_counts().sort_index(), "\n")
    
    
    new_validation_data_scaled = combined_eval.iloc[val_idx].copy()
    new_test_data_scaled = combined_eval.iloc[test_idx].copy()
    print(f"\n Validation Split Label Proportion")
    print(new_validation_data_scaled.iloc[:, -2].value_counts(normalize=True).sort_index())
    print("Test Split Label Proportion")
    print(new_test_data_scaled.iloc[:, -2].value_counts(normalize=True).sort_index())

    if verbose:
        print_post_standardize_stats(training_data_scaled, "TRAIN", acc_gyro_cols)
        print_post_standardize_stats(new_validation_data_scaled, "VAL", acc_gyro_cols)
        print_post_standardize_stats(new_test_data_scaled, "TEST", acc_gyro_cols)

        print_top_shifted_channels(training_data_scaled, acc_gyro_cols, "TRAIN")
        print_top_shifted_channels(new_validation_data_scaled, acc_gyro_cols, "VAL")
        print_top_shifted_channels(new_test_data_scaled, acc_gyro_cols, "TEST")

        print_outlier_rates(training_data_scaled, acc_gyro_cols, "TRAIN")
        print_outlier_rates(new_validation_data_scaled, acc_gyro_cols, "VAL")
        print_outlier_rates(new_test_data_scaled, acc_gyro_cols, "TEST")
    ############################################
    
    train_ds = downsample_pam(training_data_scaled, group_id="group_id", factor=3)
    val_ds   = downsample_pam(new_validation_data_scaled, group_id="group_id", factor=3)
    test_ds  = downsample_pam(new_test_data_scaled, group_id="group_id", factor=3)

    # ----- Sliding Windows -----
    # TRAIN
    X_all, y_all = [], []
    for _, gdf in train_ds.groupby("group_id", sort=False):
        gdf = gdf.drop(columns=["group_id"]).reset_index(drop=True)
        X_features = gdf.iloc[:, :-1]
        y_labels = gdf.iloc[:, -1]
        Xw, yw = sliding_window(X_features, y_labels, window_size=160, stride=40)
        X_all.append(Xw)
        y_all.append(yw)
    
    X_windows = np.concatenate(X_all, axis=0).astype(np.float32)
    y_windows = np.concatenate(y_all, axis=0).astype(np.int64)
    
    
    # VALIDATION
    X_val_all, y_val_all = [], []
    for _, gdf in val_ds.groupby("group_id", sort=False):
        gdf = gdf.drop(columns=["group_id"]).reset_index(drop=True)
        X_val_features = gdf.iloc[:, :-1]
        y_val_labels = gdf.iloc[:, -1]
        Xw_val, yw_val = sliding_window(X_val_features, y_val_labels, window_size=160, stride=40)
        X_val_all.append(Xw_val)
        y_val_all.append(yw_val)
    
    X_validation_windows = np.concatenate(X_val_all, axis=0).astype(np.float32)
    y_validation_windows = np.concatenate(y_val_all, axis=0).astype(np.int64)
    
    
    # TEST
    X_test_all, y_test_all = [], []
    for _, gdf in test_ds.groupby("group_id", sort=False):
        gdf = gdf.drop(columns=["group_id"]).reset_index(drop=True)
        X_test_features = gdf.iloc[:, :-1]
        y_test_labels = gdf.iloc[:, -1]
        Xw_test, yw_test = sliding_window(X_test_features, y_test_labels, window_size=160, stride=40)
        X_test_all.append(Xw_test)
        y_test_all.append(yw_test)
    
    X_test_windows = np.concatenate(X_test_all, axis=0).astype(np.float32)
    y_test_windows = np.concatenate(y_test_all, axis=0).astype(np.int64)

    #MAG per window demeaning
    X_windows = mag_data_rotation_pam(X_windows)
    X_validation_windows = mag_data_rotation_pam(X_validation_windows)
    X_test_windows = mag_data_rotation_pam(X_test_windows)

    if verbose:
        print_mag_rotation_stats(X_windows, "TRAIN")
        print_mag_rotation_stats(X_validation_windows, "VAL")
        print_mag_rotation_stats(X_test_windows, "TEST")
    
        print_window_label_distribution(y_windows, "TRAIN")
        print_window_label_distribution(y_validation_windows, "VAL")
        print_window_label_distribution(y_test_windows, "TEST")
    
    if verbose:
        print(f"{'-'*90}")
        print(f"Training (windows): {X_windows.shape}. Training Labels {y_windows.shape}")
        print(f"Validation (windows): {X_validation_windows.shape}. Validation Labels {y_validation_windows.shape}")
        print(f"Test (windows): {X_test_windows.shape}. Test Labels {y_test_windows.shape}")
    return X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows

def make_loaders_PAM(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator, verbose=False) -> tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    '''
    Creates DataLoaders for training, validation and test sets.
    Called once per seed — generator ensures reproducible shuffling.

    Args:
    X_windows, y_windows: training windows and labels
    X_validation_windows, y_validation_windows: validation windows and labels
    X_test_windows, y_test_windows: test windows and labels
    generator: seeded for reproducibility
    verbose: print class counts and batch shapes
    '''
    # ----- LabelEncoder, Transform, DataLoader -----
    label_encoder = fit_labelencoder(X_windows, y_windows)
    training_dataset = Dataset_HAR(X_windows, y_windows, label_encoder=label_encoder)
    validation_dataset = Dataset_HAR(X_validation_windows, y_validation_windows, label_encoder=label_encoder)
    test_dataset = Dataset_HAR(X_test_windows, y_test_windows, label_encoder=label_encoder)

    train_loader = DataLoader(training_dataset, batch_size = 64, shuffle = True, generator = generator, num_workers = 4, pin_memory = True, persistent_workers = True)
    val_loader = DataLoader(validation_dataset, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True, persistent_workers = True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 4, pin_memory = True, persistent_workers = True)

    # ----- Samples per label checking and batch size -----
    if verbose:
        label_to_name = {0: "LIE", 1: "SIT", 2: "STAND", 3: "WALK"}
        class_counts = {name: 0 for name in label_to_name.values()}
        for _, y_batch in train_loader:
            for label in y_batch:
                label_idx = label.item()
                class_name = label_to_name[label_idx]
                class_counts[class_name] += 1

        print(f"{'-'*90}")
        print("Training set class distribution:--")
        print(class_counts)
        print(f"{'-'*90}")
    return train_loader, val_loader, test_loader, label_encoder    

