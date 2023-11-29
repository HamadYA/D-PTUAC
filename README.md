
# D-PTUAC: A Drone-Person Tracking in Uniform Appearance Crowd: A New Dataset

## Introduction
- This is the official repository of D-PTUAC.
- Project website: https://d-ptuac.github.io/
- **Accepted in Scientific Data**
-Acronyms and Full Forms are shown in the table below.

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

- Samples of the proposed Drone-Person Tracking in Uniform Appearance Crowd (D-PTUAC) dataset showcases challenging attributes (IV, BC, UF, OCC, PV, MB, FM, OV, AAC, ARC, DEF, ROT, CD, SV, ST, LT) through: Row 1: RGB sample images, Row 2: Depth sample images, and Row 3: Segmentation masks sample images. Columns in the figure display samples with multiple attributes. Column (a) has IV, BC, and UF, column (b) has OCC, BC, IV, and UF, column (c) has IV, BC, and UF, column (d) has OCC, BC, IV, UF, and PV, column (e) DEF, LR, BC, IV, and UF, column (f) has MB, and FM, and column (g) has OV. These images emphasize the importance of developing robust drone-person tracking methods.

![D-PTUAC](/images/samples.png)

## Reproducability
- To reproduce the VOT results:
    
    1) Pretrained models:
    
    a) Download [pretrained tracking results](https://github.com/HamadYA/D-PTUAC/releases/download/v2/tracking_results_pretrained.zip), rename it to **tracking_results**, and place it in [~/D-PTUAC/pytracking/pytracking/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking).
    
    b) Downlaoad, place it in [~/D-PTUAC/pytracking/pytracking/notebooks/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking/notebooks), and run the pretrained evaluation codes [scentific_data_pretrained.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_pretrained.ipynb)[scentific_data_pretrained_scenarios_P1.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_pretrained_scenarios_P1.ipynb), and [scentific_data_pretrained_scenarios_P2.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_pretrained_scenarios_P2.ipynb).

    2) Finetuned models:
    
    a) Download [finetuned tracking results](https://github.com/HamadYA/D-PTUAC/releases/download/v1/tracking_results_finetuned.zip), rename it to **tracking_results**, and place it in [~/D-PTUAC/pytracking/pytracking/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking).
    
    b) Downlaoad, place it in [~/D-PTUAC/pytracking/pytracking/notebooks/](https://github.com/HamadYA/D-PTUAC/tree/main/pytracking/pytracking/notebooks), and run the finetuned evaluation code [scentific_data_finetuned.ipynb](https://github.com/HamadYA/D-PTUAC/releases/download/v3/scentific_data_finetuned.ipynb).

## Installation
- We include the installation guide in install.md for each Visual Object Tracker.

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

    2) Rename got10k.py in ~/Visual Object Tracker/lib/train/dataset/got10k.py to got10k_original.py and change it by got10k.py in [files](https://github.com/HamadYA/D-PTUAC/releases/tag/v5).


## Testing
- AiATrack



## Training
- 


