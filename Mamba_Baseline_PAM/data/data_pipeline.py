import pandas as pd
from pathlib import Path 
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter


def load_pamap2(files_path, add_group_id = False) -> pd.DataFrame:
    '''
    Read multiple .dat files from the PAMAP2 dataset and vertically concatenate
    them into a single DataFrame.

    If add_group_id=True, append a last column called 'group_id'
    with the subject id: e.g. 101, 102, ..., 109.
    '''
    dataframes = []
    row_counts = 0

    for file in files_path:
        df = pd.read_csv(file, sep=r'\s+', header=None, engine='python', na_values='NaN')
        df = df.astype(np.float32)
        if add_group_id:
            subject_id = Path(file).stem[-3:]   # subject101 -> 101
            df["group_id"] = subject_id         # added as LAST column

        dataframes.append(df)
        row_counts += len(df)

    complete_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

    assert len(complete_dataframe) == row_counts, (
        f"Row mismatch: complete_dataframe={len(complete_dataframe)} vs row_counts={row_counts}"
    )

    return complete_dataframe    


def filter_loco_pam(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Filter PAMAP2 data for the locomotion setup used in this project.
    - take activityID (col 1) as label
    - keep only locomotion labels 1, 2, 3, 4
    - drop timestamp, activityID, heart rate
    - remove temperature, second accelerometer, and orientation
    - keep group_id untouched as the LAST column if it exists

    Output:
    - training:  features + Label
    - val/test:  features + Label + group_id
    '''
    dataset = dataframe.copy()

    group_id = None
    if "group_id" in dataset.columns:
        group_id = dataset["group_id"].copy()
        dataset = dataset.drop(columns=["group_id"])
    
    keep_rows = dataset.iloc[:, 1].isin([1, 2, 3, 4])
    dataset = dataset.loc[keep_rows].copy()

    if group_id is not None:
        group_id = group_id.loc[keep_rows].copy()

    label = dataset.iloc[:, 1].copy()
    dataset = dataset.drop(columns=[0, 1, 2])  # drop timestamp, label, heart rate

    imu_chunk = 17
    temp_cols = [col for col in range(3, dataset.shape[1], imu_chunk)]  # remove temp
    acc_second_cols = [col + axis for col in range(7, dataset.shape[1], imu_chunk) for axis in range(3)]  # remove second acc
    ori_cols = [col + axis for col in range(16, dataset.shape[1], imu_chunk) for axis in range(4)]  # remove orientation

    dataset = dataset.drop(columns=temp_cols + acc_second_cols + ori_cols)
    dataset["Label"] = label.values                # label last col
    dataset.columns = range(len(dataset.columns))  # reset feature cols

    if group_id is not None:
        dataset["group_id"] = group_id.values      # group_id added to LAST col
    return dataset

def interpolation_pam(dataframe: pd.DataFrame, max_gap: int = 30) -> pd.DataFrame:
    '''
    Interpolate short NaN chunks in sensor columns only.
    Chunks with length <= max_gap are linearly interpolated.
    Longer chunks are left as NaN and removed at the end.
    ''' 
    df = dataframe.copy()
    group_id = df.columns[-1] == "group_id"
    sensor_cols = df.columns[:-2] if group_id else df.columns[:-1]    
    sensor_data = df.loc[:, sensor_cols]
    
    nan_mask = sensor_data.isna().any(axis=1).to_numpy()
    nan_positions = np.where(nan_mask)[0]
    if len(nan_positions) == 0:
        return df

    gaps = np.diff(nan_positions)
    seg_starts = np.insert(np.where(gaps > 1)[0] + 1, 0, 0) # start of segment nan
    seg_ends = np.append(np.where(gaps > 1)[0], len(nan_positions) - 1) # end of segment nan

    for s, e in zip(seg_starts, seg_ends):
        start = nan_positions[s]
        end = nan_positions[e]
        seg_len = end - start + 1

        if seg_len <= max_gap:
            rows = df.index[max(start - 1, 0): min(end + 2, len(df))]
            interp = df.loc[rows, sensor_data.columns].interpolate(method="linear", axis=0, limit_area="inside")
            df.loc[rows, sensor_data.columns] = interp

    sensor_data_after = df.iloc[:, :-2] if df.columns[-1] == "group_id" else df.iloc[:, :-1]
    keep_rows = ~sensor_data_after.isna().any(axis=1)
    return df.loc[keep_rows].reset_index(drop=True)

def remove_zero_label_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove rows where the label is 0.
    Works for:
      - train: label in last column
      - val/test: label in second-to-last column if last column is 'group_id'
    '''
    #mask_nans = dataframe.iloc[:, :].isna().any(axis=1)
    label_col_idx = -2 if dataframe.columns[-1] == "group_id" else -1
    zero_label = dataframe.iloc[:, label_col_idx] == 0
    #print(f"Number of NaN rows in the data: {mask_nans.sum()} | Number of 0 labels: {zero_label.sum()}")
    return dataframe[~(zero_label)]

def remove_all_nan_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove any row with any NaN value. 
    (No interpolation done in the dataset later)
    '''
    mask_nans = dataframe.isna().any(axis = 1)
    #print(f"Number of NaN rows in the data: {mask_nans.sum()}")
    return dataframe.loc[~mask_nans].copy()
    
def incomplete_labeled_rows(dataframe: pd.DataFrame) -> int:
    '''
    Count rows with at least one NaN in sensor columns only.
    Excludes:
      - label column in train
      - label + group_id in val/test
    '''
    sensor_data = dataframe.iloc[:, :-2] if dataframe.columns[-1] == "group_id" else dataframe.iloc[:, :-1]
    incomplete_rows = sensor_data.isna().any(axis=1)
    return int(incomplete_rows.sum())

def divide_features_labels(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = dataframe.iloc[:, :-1]
    labels = dataframe.iloc[:, -1]
    return features, labels


def acc_data_scaling_pam (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    PAMAP2 acceleration is in m/s^2.
    Divide by 9.81 to convert to g.
    '''
    dataframe_ = dataframe.copy()
    acc_cols = [axi + offset for offset in range(0, 27, 9) for axi in range(3)]
    dataframe_.iloc[:, acc_cols] /= 9.81
    return dataframe_ 

def mag_data_norm_pam(dataframe: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    '''
    Normalize each magnetometer 3D vector by its magnitude.
    This removes absolute field strength and keeps directional information.
    '''
    dataframe_ = dataframe.copy()

    for offset in range(6, 27, 9):
        mag_x = dataframe_.iloc[:, offset]
        mag_y = dataframe_.iloc[:, offset + 1]
        mag_z = dataframe_.iloc[:, offset + 2]

        magnitude = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        magnitude = magnitude.clip(lower = eps)

        for axi in range(3):
            dataframe_.iloc[:, axi + offset] = dataframe_.iloc[:, axi + offset] / magnitude
    return dataframe_
    
def sliding_window(X, y, window_size: int, stride: int, return_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Row-wise sliding windows and assign a label policy: majority vote.
    Returns: features_windows: [#windows, window_size, #features]
             labels_windows: [#windows]
    '''
    
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_series = isinstance(y, (pd.Series, pd.DataFrame))
    X_values = X.values if X_is_df else np.asarray(X)
    y_values = y.values.ravel() if y_is_series else np.asarray(y).ravel()

    num_rows, num_feats = X_values.shape
    X_windows = []
    y_windows = []
    window_count = 0
    # Slide Window
    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        Xw = X_values[start:end, :]
        yw = y_values[start:end]

        win_label = Counter(yw).most_common(1)[0][0] # counts times each label appears. Take the tuple. Take the label (first position)
        X_windows.append(Xw)
        y_windows.append(int(win_label))
        window_count += 1

    #print(f"Number of windows created: {window_count}")

    # Stack Outputs
    if len(X_windows) == 0:
        if return_numpy:
            return np.empty((0, window_size, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)
        else:
            return [], []

    if return_numpy:
        X_windows = np.stack(X_windows).astype(np.float32)  # [W,L,F]
        y_windows = np.asarray(y_windows, dtype=np.int64)   # [W]
    
    return X_windows, y_windows


def downsample_pam(dataframe: pd.DataFrame, group_id: str = "group_id", factor: int = 3) -> pd.DataFrame:
    """
    Downsample PAMAP2 from 100 Hz to ~33.3 Hz BEFORE sliding windows.

    Assumes dataframe = features + Label + group_id
    Keeps group_id as last column.
    """
    if group_id not in dataframe.columns:
        raise ValueError(f"'{group_id}' must be present.")

    out_groups = []

    for _, gdf in dataframe.groupby(group_id, sort=False):
        gdf = gdf.reset_index(drop=True)

        feature_cols = gdf.columns[:-2]
        label_col = gdf.columns[-2]

        X = gdf.loc[:, feature_cols].to_numpy(dtype=np.float32)
        y = gdf.loc[:, label_col].to_numpy()

        rows = []
        for start in range(0, len(gdf) - factor + 1, factor):
            end = start + factor

            x_block = X[start:end]
            y_block = y[start:end]

            x_new = x_block.mean(axis=0)  # simple and clean
            y_new = Counter(y_block).most_common(1)[0][0]

            rows.append(list(x_new) + [y_new, gdf[group_id].iloc[0]])

        down_gdf = pd.DataFrame(rows, columns=list(feature_cols) + [label_col, group_id])
        out_groups.append(down_gdf)

    return pd.concat(out_groups, axis=0, ignore_index=True)


    
    
'''def resample_windows_pool(X_windows: np.ndarray, target_len: int = 150, method: str = "median") -> np.ndarray:

    Resample a batch of windows from [N, old_len, F] to [N, target_len, F]
    using median or mean pooling.
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'")

    num_windows, long_freq_length, features = X_windows.shape
    edges = np.round(np.linspace(0, long_freq_length, target_len + 1)).astype(int) # create bins, 151 edges defining bins

    X_resampled = np.empty((num_windows, target_len, features), dtype=np.float32) # pre-allocation of the output array
 
    for w in range(num_windows):
        for i in range(target_len):
            start = edges[i]
            end = edges[i + 1]

            if end <= start:
                end = min(start + 1, long_freq_length)

            chunk = X_windows[w, start:end, :]

            if method == "median":
                X_resampled[w, i, :] = np.median(chunk, axis=0)
            else:
                X_resampled[w, i, :] = np.mean(chunk, axis=0)
    return X_resampled

def create_pamap_windows_from_split(dataframe: pd.DataFrame, group_id: str = "group_id", window_size: int = 500, stride: int = 250, target_len: int = 150, resample_method: str = "median") -> Tuple[np.ndarray, np.ndarray]:

    For one split dataframe (train / val / test):
    1) split by group_id
    2) remove group_id inside each group
    3) apply normal sliding_window with majority vote
    4) resample each window from 500 -> 150

    Returns:
        X_final: [N, 150, F]
        y_final: [N]

    if group_id not in dataframe.columns:
        raise ValueError(f"'{group_id}' must be present before windowing.")

    X_all = []
    y_all = []

    for _, group_df in dataframe.groupby(group_id, sort=False):
        # remove group_id
        group_df = group_df.drop(columns=[group_id]).reset_index(drop=True)
        X_group, y_group = divide_features_labels(group_df)

        # raw PAMAP2 windows at 100 Hz
        Xw, yw = sliding_window(X_group, y_group, window_size = window_size, stride = stride, return_numpy = True)

        if len(Xw) == 0:
            continue

        # resample each window: 500 -> 150
        Xw_resampled = resample_windows_pool(Xw, target_len = target_len, method = resample_method)
        X_all.append(Xw_resampled)
        y_all.append(yw)

    if len(X_all) == 0:
        num_feats = dataframe.drop(columns=[group_id]).shape[1] - 1  # minus label column
        return np.empty((0, target_len, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_final = np.concatenate(X_all, axis = 0).astype(np.float32)
    y_final = np.concatenate(y_all, axis = 0).astype(np.int64)

    return X_final, y_final'''
    
        
def mag_data_rotation_pam(X_windows) -> np.ndarray:
    X_w = X_windows.copy()
    mag_cols = [axis + offset for offset in range(6, 27, 9) for axis in range(3)]
    for channel in mag_cols:
        mean_val = X_w[:, :, channel].mean(axis=1, keepdims=True)
        X_w[:, :, channel] -= mean_val
    return X_w    




