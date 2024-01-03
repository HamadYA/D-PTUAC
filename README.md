
# D-PTUAC: A Drone-Person Tracking in Uniform Appearance Crowd: A New Dataset
![D-PTUAC](/images/samples.png)
- Samples of the proposed Drone-Person Tracking in Uniform Appearance Crowd (D-PTUAC) dataset showcases challenging attributes (IV, BC, UF, OCC, PV, MB, FM, OV, AAC, ARC, DEF, ROT, CD, SV, ST, LT) through: Row 1: RGB sample images, Row 2: Depth sample images, and Row 3: Segmentation masks sample images. Columns in the figure display samples with multiple attributes. Column (a) has IV, BC, and UF, column (b) has OCC, BC, IV, and UF, column (c) has IV, BC, and UF, column (d) has OCC, BC, IV, UF, and PV, column (e) DEF, LR, BC, IV, and UF, column (f) has MB, and FM, and column (g) has OV. These images emphasize the importance of developing robust drone-person tracking methods.

## Introduction
- This is the official repository of D-PTUAC.
- Paper Link: https://www.nature.com/articles/s41597-023-02810-y
- Project Website: https://d-ptuac.github.io/
- Demo Video: https://www.youtube.com/watch?v=ivrBrjGawm4
- **Accepted in Scientific Data**
- Acronyms and Full Forms are shown in the table below.

| Acronym | Full Form | Acronym      | Full Form   |
  | -------------- | ----- | -------- | -------- 
| UF | Uniformity |  PV | Pose Variation|
| AAC | Abrupt Appearance Change | ROT | Rotation |
| ARC | Aspect Ratio Change | SV | Scale Variation |
| BC | Background Clutter | SS | Surveillance Settings |
| DEF | Deformation | CD | Compact Density |
| IV | Illumination Variation | MD | Medium Density |
| FM | Fast Motion | SD | Sparse Density |
| LT | Long-Term | S1 | Scenario 1 |
| ST | Short-Term | S2 | Scenario 2 | 
| LR | Low Resolution | S3 | Scenario 3 |
| MB | Motion Blur | S4 | Scenario 4 |
| OV | Out of View | S5 | Scenario 5 |
| OCC | Occlusion | FOC | Full Occlusion |
| POC | Partial Occlusion | VOT | Visual Object Tracking |

## Reproducability
- To reproduce the VOT results:
    
    1) Pretrained models:
    
    a) Download [pretrained tracking results](https://github.com/HamadYA/D-PTUAC/releases/download/v2/tracking_results_pretrained.zip), rename it to **tracking_results**, and place it in [~/D-PTUAC/pytracking/pytracking/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking).
    
    b) Download, place it in [~/D-PTUAC/pytracking/pytracking/notebooks/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking/notebooks), and run the pretrained evaluation codes [scentific_data_pretrained.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_pretrained.ipynb)[scentific_data_pretrained_scenarios_P1.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_pretrained_scenarios_P1.ipynb), and [scentific_data_pretrained_scenarios_P2.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_pretrained_scenarios_P2.ipynb).

    2) Finetuned models:
    
    a) Download [finetuned tracking results](https://github.com/HamadYA/D-PTUAC/releases/download/v1/tracking_results_finetuned.zip), rename it to **tracking_results**, and place it in [~/D-PTUAC/pytracking/pytracking/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking).
    
    b) Downlaoad, place it in [~/D-PTUAC/pytracking/pytracking/notebooks/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking/notebooks), and run the finetuned evaluation code [scentific_data_finetuned.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_finetuned.ipynb).

## Installation
- We include the installation guide in [link](https://github.com/HamadYA/D-PTUAC/releases/tag/v8) for some Visual Object Trackers.

## Folder Structure
- Download pretrained weights of all models and place them in the structure shown below.

```sh
    .               
    ├── pytracking
    │   ├── checkpoints
    |   |   ├── ltr
    |   |   |   ├── 'model'
    |   |   |   |    ├── 'parameter'
    |   |   |   |   |   ├── 'weights.pth.tar'
    |   |   |   |   |   ├── 'weights.pth'
    │   ├── pytacking 
    |   |   ├── networks
    |   |   |   ├── atom_default.pth 
    |   |   |   ├── dimp18.pth
    |   |   |   ├── dimp50.pth
    |   |   |   ├── prdimp18.pth.tar
    |   |   |   ├── prdimp50.pth.tar
    |   |   |   ├── e2e_mask_rcnn_R_50_FPN_1x_converted.pkl
    |   |   |   ├── keep_track.pth.tar
    |   |   |   ├── kys.pth
    |   |   |   ├── lwl_boxinit.pth
    |   |   |   ├── lwl_stage2.pth
    |   |   |   ├── rts50.pth
    |   |   |   ├── sta.pth.tar
    |   |   |   ├── super_dimp.pth.tar
    |   |   |   ├── super_dimp_simple.pth.tar
    |   |   |   ├── tomp50.pth.tar
    |   |   |   ├── tomp101.pth.tar
    |   |   |   ├── resnet18_vggmconv1
    |   |   |   |    ├── resnet18_vggmconv1.pth
    |   │   ├── pretrained_networks
    |   |   |   |    ├── super_dimp_simple.pth.tar
    ├── AiATrack 
    |   ├── checkpoints
    |   |   ├── train
    |   |   |   ├── aiatrack
    |   |   |   |   ├── baseline
    |   |   |   |   |   ├── AIATRACK_ep0500.pth.tar
    ├── DropTrack 
    |   ├── checkpoints
    |   |   ├── train
    |   |   |   ├── ostrack
    |   |   |   |   ├── vitb_384_mae_ce_32x4_ep300
    |   |   |   |   |   ├── OSTrack_ep0500.pth.tar
    |   ├── pretrained_models
    |   |   ├── k700_800E_checkpoint_final.pth
    ├── Stark
    |   ├── checkpoints
    |   |   ├── train
    |   |   |   ├── stark_s
    |   |   |   |   ├── baseline
    |   |   |   |   |   ├── STARKS_ep0050.pth.tar
    |   |   |   ├── stark_st1
    |   |   |   |   ├── baseline
    |   |   |   |   |   ├── STARKST_ep0050.pth.tar
    |   |   |   ├── stark_st2
    |   |   |   |   ├── baseline
    |   |   |   |   |   ├── STARKST_ep0050.pth.tar
    ├── MKDNet
    |   ├── checkpoints
    |   |   ├── ltr
    |   |   |   ├── dimp
    |   |   |   |   ├── super_dimp
    |   |   |   |   |   ├── DiMPnet_ep0030.pth.tar
    |   ├── pytracking
    |   |   ├── networks
    |   |   |   ├── super_dimp.pth.tar
    |   |   |   ├── DiMPnet_ep0030.pth.tar
    ├── SeqTrack 
    |   ├── checkpoints
    |   |   ├── train
    |   |   |   ├── seqtrack
    |   |   |   |   ├── seqtrack_b256
    |   |   |   |   |   ├── SEQTRACK_ep0500.pth.tar
    |   |   |   |   ├── seqtrack_b384
    |   |   |   |   |   ├── SEQTRACK_ep0500.pth.tar
    |   |   |   |   ├── seqtrack_l256
    |   |   |   |   |   ├── SEQTRACK_ep0500.pth.tar
    |   |   |   |   ├── seqtrack_b384
    |   |   |   |   |   ├── SEQTRACK_ep0010.pth.tar
    ├── TransformerTrack
    |   ├── pytracking
    |   |   ├── networks
    |   |   |   ├── trdimp_net.pth.tar
    |   ├── checkpoints
    |   |   ├── ltr  
    |   |   |   ├── dimp 
    |   |   |   |   ├── transformer_dimp
    |   |   |   |   |   ├── DiMPnet_ep0010.pth.tar
    ├── NeighborTrack
    |   ├── trackers
    |   |   ├── ostrack
    |   |   |   ├── pretrained_models
    |   |   |   ├── output
    |   |   |   |   ├── checkpoints
    |   |   |   |   |   ├── train
    |   |   |   |   |   |   ├── ostrack
    |   |   |   |   |   |   |   ├── vitb_384_mae_ce_32x4_ep300_neihbor
    |   |   |   |   |   |   |   |   ├── OSTrack_ep0300.pth.tar
    ├── TrTr
    |   ├── networks
    |   |   ├── trtr_resnet50.pth
    ├── TransT
    |   ├── pytracking
    |   |   ├── networks
    |   |   |   ├── transt.pth
    ├── ettrack
    |   ├── pytracking
    |   |   ├── networks
    |   |   |   ├── 
    ├── MixFormer 
    |   ├── checkpoints
    |   |   ├── train
    |   |   |   ├── 
    |   |   |   |   ├── baseline
    |   |   |   |   |   ├── 
    ├── TATrack
    |   ├── checkpoints
    |   |   ├── train
    |   |   |   ├── 
    |   |   |   |   ├── baseline
    |   |   |   |   |   ├── 
    ├── SLTtrack
    |   ├── checkpoints
    |   |   ├── ltr
    |   |   |   ├── slt_trdimp
    |   |   |   |   ├── slt_trdimp
    |   |   |   |   |   ├── DiMPnet_ep0010.pth.tar
    

```
## Data Loader
- As our dataset follow got10k dataset format, for simplicity, we use got10k dataset loader and make the following changes:

  - For the following Visual Object Trackers: OSTrack, MixFormer, Stark, AiATrack, NeighborTrack, SeqTrack, and DropTrack.
  
    1) Replace ~/Visual Object Tracker/lib/train/data_specs/*
    a) got10k_train_full_split
    b) got10k_train_split
    c) got10k_val_split

    by these [files](https://github.com/HamadYA/D-PTUAC/releases/tag/v5).

    2) For Training:
    a) Rename got10k.py in ~/Visual Object Tracker/lib/train/dataset/got10k.py to got10k_original.py and change it by lib_got10k.py (change its name to got10k.py) in [files](https://github.com/HamadYA/D-PTUAC/releases/tag/v5).
    b) Edit or replace experiments ~/Visual Object Tracker/experiments/Visual Object Tracker/baseline.yaml by these [files](https://github.com/HamadYA/D-PTUAC/releases/tag/v6).

    3) There is no need to edit the testing data loader as our data follows the same format as got10k dataset.

- For the following Visual Object Trackers: pytracking, TransformerTrack, SLTtrack, ettrack, TransT, and MKDNet.
  
    1) Replace ~/Visual Object Tracker/ltr/data_specs/*
    a) got10k_train_full_split
    b) got10k_train_split
    c) got10k_val_split

    by these [files](https://github.com/HamadYA/D-PTUAC/releases/tag/v5).

    2) For Training:
    a) Rename got10k.py in ~/Visual Object Tracker/ltr/dataset/got10k.py to got10k_original.py and change it by py_got10k.py (change its name to got10k.py) in [files](https://github.com/HamadYA/D-PTUAC/releases/tag/v5).
    b) Edit or replace experiments ~/Visual Object Tracker/ltr/train_settings/Visual Object Tracker.py by these [files](https://github.com/HamadYA/D-PTUAC/releases/tag/v7) with each correspond to each Visual Object Tracker.

    3) There is no need to edit the testing data loader as our data follows the same format as got10k dataset.

## Testing
- AiATrack
```sh
    cd ~/AiATrack
    # Edit paths in ~/AiATrack/lib/test/evaluation/local.py and ~/AiATrack/lib/train/admin/local.py
    python tracking/lib/test.py --param baseline --dataset got10k_test
  ```

- DropTrack
  ```sh
    cd ~/DropTrack
    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
    # Edit paths in ~/DropTrack/lib/test/evaluation/local.py and ~/DropTrack/lib/train/admin/local.py
    # Edit ~/DropTrack/lib/models/ostrack/ostrack.py by adding pretrained_path line 97 to the correct path which is:
    python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset got10k_test --threads 1 --num_gpus 1
  ```

- Stark
  ```sh
    cd ~/Stark
    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
    # Edit paths in ~/Stark/lib/test/evaluation/local.py and ~/Stark/lib/train/admin/local.py
    python tracking/test.py stark_st baseline_got10k_only --dataset got10k_test --threads 1
  ```

- SeqTrack
  ```sh
    cd ~/SeqTrack
    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
    # Edit paths in ~/SeqTrack/lib/test/evaluation/local.py and ~/SeqTrack/lib/train/admin/local.py
    python tracking/test.py seqtrack seqtrack_b256 --dataset got10k_test --threads 1
  ```

- OSTrack
  ```sh
    cd ~/OSTrack
    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
    # Edit paths in ~/OSTrack/lib/test/evaluation/local.py and ~/OSTrack/lib/train/admin/local.py
    python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset got10k_test --threads 1 --num_gpus 1
  ```

- NeighborTrack
  ```sh
    cd ~/NeighborTrack/trackers/ostrack/
    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
    # Edit paths in ~/NeighborTrack/trackers/ostrack/lib/test/evaluation/local.py and ~/NeighborTrack/trackers/ostrack/lib/train/admin/local.py
    python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300_neighbor --dataset got10k_test --threads 1 --num_gpus 1 --neighbor 1
  ```

- MixFormer
  ```sh
    cd ~/MixFormer
    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
    # Edit paths in ~/MixFormer/lib/test/evaluation/local.py and ~/MixFormer/lib/train/admin/local.py
    # Edit test_mixformer_*.sh and then run them
    bash tracking/test_mixformer_cvt.sh
  ```

- pytracking
  ```sh
    cd ~/pytracking
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

    # Edit paths in ~/pytracking/pytracking/evaluation/local.py and ~/pytracking/ltr/admin/local.py
    # You can run any tracker that the pytracking library provides such as the below command:
    python pytracking/run_tracker tomp tomp50 --dataset_name got10k_test
  ```

- DeT
  ```sh
    cd ~/DeT
    # You need to download the dataset with monocular depth images which can be found in [1](https://kuacae-my.sharepoint.com/:f:/g/personal/100061914_ku_ac_ae/EmraqT_5nCNHsIyBNUdHDbkBq22XAUudYv7XB7v1zgeBKw?e=o3mxit) and [depth](https://doi.org/10.6084/m9.figshare.24081597.v1) (please cite)
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

    # Edit paths in ~/MKDNet/pytracking/evaluation/local.py and ~/MKDNet/ltr/admin/local.py
    python pytracking/run_tracker dimp DeT_DiMP50_Mean --dataset_name got10k_test
  ```

- MKDNet
  ```sh
    cd ~/MKDNet
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

    # Edit paths in ~/MKDNet/pytracking/evaluation/local.py and ~/MKDNet/ltr/admin/local.py
    python pytracking/run_tracker dimp super_dimp --dataset_name got10k_test
  ```

- TransformerTrack
  ```sh
    cd ~/TransformerTrack
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

    # Edit paths in ~/TransformerTrack/pytracking/evaluation/local.py and ~/TransformerTrack/ltr/admin/local.py
    python pytracking/run_tracker trdimp trdimp --dataset_name got10k_test
  ```

- ettrack
  ```sh
    cd ~/ettrack
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

    # Edit paths in ~/ettrack/pytracking/evaluation/local.py and ~/ettrack/ltr/admin/local.py
    python pytracking/run_tracker et_tracker et_tracker --dataset_name got10k_test
  ```

- SLTtrack
  ```sh
    cd ~/SLTtrack
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

    # Edit paths in ~/SLTtrack/pytracking/evaluation/local.py and ~/SLTtrack/ltr/admin/local.py
    python pytracking/run_tracker slt_trdimp slt_trdimp --dataset_name got10k_test
  ```

- TransT
  ```sh
    cd ~/TransT
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

    # Edit paths in ~/TransT/pytracking/evaluation/local.py and ~/TransT/ltr/admin/local.py
    python pytracking/run_tracker transt transt --dataset_name got10k_test
  ```

- TrTr
  ```sh
    # Place the dataset under TrTr/benchmark/dataset/GOT-10k
    cd ~/TrTr
    cd benchmark
    # Edit and run
    python test.py --cfg_file ../parameters/experiment/got10k/offline.yaml
  ```

  - TATrack
  ```sh
    cd ~/TATrack
    # Edit and run
    python main/test.py --config experiments/tatrack/test/base/got.yaml
  ```

## Training
- AiATrack
```sh
    cd ~/AiATrack
    # Edit paths in ~/AiATrack/lib/test/evaluation/local.py and ~/AiATrack/lib/train/admin/local.py
    # After preparing the environment and following the data loader guide, you can run:
    python tracking/train.py --mode single --nproc 1
  ```

- DropTrack
```sh
    cd ~/DropTrack
    # Edit paths in ~/DropTrack/lib/test/evaluation/local.py and ~/DropTrack/lib/train/admin/local.py
    # After preparing the environment and following the data loader guide, you can run:
    python tracking/train.py --script ostrack --config vitb_384_mae_ce_32x4_ep300 --save_dir save_path  --mode single --nproc_per_node 1 --use_lmdb 0 --use_wandb 0
  ```

- Stark
```sh
    cd ~/Stark
    # Edit paths in ~/Stark/lib/test/evaluation/local.py and ~/Stark/lib/train/admin/local.py
    # After preparing the environment and following the data loader guide, you can run:
    python tracking/train.py --script stark_st1 --config baseline --save_dir . --mode single --nproc_per_node 1  # STARK-ST50 Stage1
    python tracking/train.py --script stark_st2 --config baseline --save_dir . --mode single --nproc_per_node 1 --script_prv stark_st1 --config_prv baseline  # STARK-ST50 Stage2
  ```

- OSTrack
```sh
    cd ~/OSTrack
    # Edit paths in ~/OSTrack/lib/test/evaluation/local.py and ~/OSTrack/lib/train/admin/local.py
    # After preparing the environment and following the data loader guide, you can run:
    python tracking/train.py --script ostrack --config vitb_384_mae_ce_32x4_ep300 --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 1
  ```

- SeqTrack
```sh
    cd ~/SeqTrack
    # Edit paths in ~/SeqTrack/lib/test/evaluation/local.py and ~/SeqTrack/lib/train/admin/local.py
    # After preparing the environment and following the data loader guide, you can run:
    python tracking/train.py --script seqtrack --config seqtrack_b256 --save_dir . --mode single
  ```

- MixFormer
```sh
    cd ~/MixFormer
    # Edit paths in ~/MixFormer/lib/test/evaluation/local.py and ~/MixFormer/lib/train/admin/local.py
    # After preparing the environment and following the data loader guide, you can run (first, edit train_mixformer_cvt.sh):
    bash tracking/train_mixformer_cvt.sh
  ```

- pytracking
```sh
    cd ~/pytracking
    # Edit paths in ~/pytracking/pytracking/evaluation/local.py and ~/pytracking/ltr/admin/local.py
    # You can train any tracker that the pytracking library provides such as the below command:
    python run_training.py dimp super_dimp
  ```

- MKDNet
```sh
    cd ~/MKDNet
    # Edit paths in ~/MKDNet/pytracking/evaluation/local.py and ~/MKDNet/ltr/admin/local.py
    # You can train any tracker that the pytracking library provides such as the below command:
    python run_training.py dimp super_dimp
  ```

- TransformerTrack
```sh
    cd ~/TransformerTrack
    # Edit paths in ~/TransformerTrack/pytracking/evaluation/local.py and ~/TransformerTrack/ltr/admin/local.py
    # You can train any tracker that the pytracking library provides such as the below command:
    python run_training.py trdimp trdimp
  ```

- SLTtrack
```sh
    cd ~/SLTtrack
    # Edit paths in ~/SLTtrack/pytracking/evaluation/local.py and ~/SLTtrack/ltr/admin/local.py
    # You can train any tracker that the pytracking library provides such as the below command:
    python run_training.py slt_trdimp slt_trdimp
  ```

- DeT
```sh
    cd ~/DeT
    # Edit paths in ~/DeT/pytracking/evaluation/local.py and ~/DeT/ltr/admin/local.py
    # You can train any tracker that the pytracking library provides such as the below command:
    python run_training.py dimp DeT_DiMP50_Mean
  ```



# Citing
  - **BibTeX**
    ```bibtex
    @article{Alansari2023_with_depth,
    author = "Mohamad Alansari",
    title = "{D-PTUAC.zip}",
    year = "2023",
    month = "9",
    url = "https://figshare.com/articles/dataset/D-PTUAC_zip/24081597",
    doi = "10.6084/m9.figshare.24081597.v1"
    }
    ```
    ```bibtex
    @article{Alansari2023,
    author = "Mohamad Alansari and Oussama Abdulhay and Sara Alansari and Sajid Javed and Abdulhadi Shoufan and Yahya Zweiri and Naoufel Werghi",
    title = "{Drone-Person Tracking in Uniform Appearance Crowd (D-PTUAC)}",
    year = "2023",
    month = "11",
    url = "https://figshare.com/articles/dataset/Drone-Person_Tracking_in_Uniform_Appearance_Crowd_D-PTUAC_/24590568",
    doi = "10.6084/m9.figshare.24590568.v2"
    }
    ```
