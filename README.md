# TFM Deep Learning — UPM



  ### Human Activity Recognition with a Mamba (SSM) backbone







  Master's thesis code comparing **supervised** vs. **self-supervised (SSL)** training of a



  Mamba-2 based encoder for Human Activity Recognition on the **PAMAP2** and **Opportunity**



  wearable-sensor datasets. Training is designed to run on a SLURM GPU cluster (A100 or H100).


  ---




  ## Repository structure


  | `Mamba_Baseline_OPP/`     | Supervised Mamba classifier — **Opportunity** (LOSO 4-fold). |



  | `Mamba_Baseline_PAM/`     | Supervised Mamba classifier — **PAMAP2** (subject-wise 4-fold). |



  | `SELFSUPERVISED_MAMBA/`   | Self-supervised pre-training + downstream fine-tuning — **Opportunity**. |



  | `SELFSUPERVISED_MAMBA_PAM/`| Self-supervised pre-training + downstream fine-tuning — **PAMAP2**. |



  | `DATASETS/`               | Raw datasets (not tracked in git — see below). 







  Each experiment folder shares the same building blocks:







  - **`Mamba_blocks.py`** — Mamba/Mamba-2 mixer blocks (`MixerModel`), wrapping `mamba_ssm`.



  - **`MambaClassificationModel.py` / `MambaSSLModel.py`** — supervised classifier / SSL encoder



    (`HARMambaConfig`: `d_model=384`, `n_layer=8`, `Mamba2`, `expand=4`, ...).



  - **`data/`** — dataset loading + preprocessing (`*_data.py`, `data_pipeline.py`, `preprocessing.py`):



    loads `.dat` files, removes null/incomplete labels, scales acc/gyro, normalizes magnetometer,



    applies sliding windows, and builds `StratifiedGroupKFold` splits.



  - **`train_*.py` / `pretrain_*.py`** — training entry points (argparse).



  - **`*.slurm` / `launch_*.sh`** — SLURM submission scripts.



  - **`logs/`, `models_pt/`, `results_json/`** — run outputs (per fold × seed).



  - **`Results_*.ipynb`** — aggregate metrics, confusion matrices and per-class F1 plots.







  The SSL pipeline runs in two stages: a self-supervised **pre-training** job, then a



  **downstream** classification job that resumes from the pretrained encoder (chained via



  SLURM `--dependency=afterok`).







  ---







  ## Datasets (place inside `DATASETS/`)




  The data loaders expect the raw datasets at "/DATASETS"



  folder at the project root, alongside the experiment folders:



  ```

  DATASETS/



  ├── PAMAP2_Dataset/



  │   └── Protocol/



  │       ├── subject101.dat ... subject108.dat



  └── OpportunityUCIDataset/



      └── dataset/



          ├── S1-ADL1.dat ... S4-ADL5.dat



          └── S1-Drill.dat ...



  ```







  - **PAMAP2** — Physical Activity Monitoring (UCI). Uses the `Protocol/` `.dat` files for



    subjects **101–108**. Folds split subjects 2/2/4 (test/val/train).



  - **Opportunity** — Opportunity Activity Recognition (UCI). Uses `S{1-4}-ADL{1-5}.dat`.



    Leave-One-Subject-Out 4-fold (ADL1–4 train, ADL5 val, held-out subject test).




  > Datasets are **not** committed.

  > Download them from the UCI ML Repository and unzip into `DATASETS/` as shown above.




  ---




  ## Installing Mamba-2







  The models depend on the official `mamba_ssm` package (CUDA GPU required — needs `nvcc`



  and a matching PyTorch CUDA build). On the cluster, inside your venv:







  ```bash



  # 1. PyTorch with CUDA (match your cluster's CUDA, e.g. cu121)



  



  # 2. Fast CUDA conv kernel + Mamba-2 SSM kernels



  pip install packaging ninja



  pip install causal-conv1d>=1.4.0



  pip install mamba-ssm            # provides mamba_ssm.modules.mamba2.Mamba2



  



  # (Triton is pulled in for the fused RMSNorm path used in Mamba_blocks.py)



  ```



  



  If you hit build errors, install with `pip install mamba-ssm --no-build-isolation`



  after loading the cluster's CUDA module (`module load cuda`).



  



  ---



  



  ## How to run



  



  All scripts assume the env is activated (`source ~/myenv/bin/activate`) and are launched



  from inside the relevant experiment folder.



  



  ### Supervised baseline (single fold)



  



  ```bash



  cd Mamba_Baseline_PAM



  sbatch train_sup_mamba_pam.slurm PAM 1      # <DATASET> <FOLD 1-4>



  # Opportunity:



  cd ../Mamba_Baseline_OPP



  sbatch train_sup_mamba.slurm OPP 1



  ```



  



  Or directly (on a GPU node): `python train_sup_mamba_pam.py --dataset PAM --fold 1`



  



  ### Self-supervised (pre-train → downstream, all 4 folds)



  



  ```bash



  cd SELFSUPERVISED_MAMBA/SSL_RUN1_ENCODER_CE_ALG1



  bash launch_pretrain_downstream_V1.sh       # submits pretrain + dependent downstream per fold



  ```



  



  Each run loops folds 1–4 and ~5 seeds; metrics land in `results_json/`, checkpoints in



  `models_pt/` (or `SSL_models_pt/`), and logs in `logs/`. Use the `Results_*.ipynb`



  notebooks to aggregate folds and reproduce the confusion-matrix / per-class-F1 figures.



  



  ---



  



  ## Outputs



  



  - `models_pt/model_<DS>_fold<F>_seed<S>.pt` — trained weights.



  - `results_json/<DS>_fold<F>_results.json` — per-seed metrics (accuracy, macro-F1, …).



  - `logs/training_<DS>_fold<F>.txt` and SLURM `%x_%j.out/.err` — training curves & job logs.
