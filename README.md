# DDHPose
The PyTorch implementation for ["Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser"](https://arxiv.org/abs/2403.04444.pdf) (AAAI 2024). 

# Dependencies
Make sure you have the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* tensorboard
You should download [MATLAB](https://www.mathworks.com/products/matlab-online.html) if you want to evaluate our model on MPI-INF-3DHP dataset.

## Datasets

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M

We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).  You can download the processed data from [here](https://drive.google.com/file/d/1FMgAf_I04GlweHMfgUKzB0CMwglxuwPe/view?usp=sharing).  `data_2d_h36m_gt.npz` is the ground truth of 2D keypoints. `data_2d_h36m_cpn_ft_h36m_dbb.npz` is the 2D keypoints obatined by [CPN](https://github.com/GengDavid/pytorch-cpn).  `data_3d_h36m.npz` is the ground truth of 3D human joints. Put them in the `./data` directory.

### MPI-INF-3DHP

We set up the MPI-INF-3DHP dataset following [D3DP](https://github.com/paTRICK-swk/D3DP). You can download the processed data from [here](https://drive.google.com/file/d/1zOM_CvLr4Ngv6Cupz1H-tt1A6bQPd_yg/view?usp=share_link). Put them in the `./data` directory. 

## Evaluating our models
You can download our pre-trained models, which are evaluated on Human3.6M (from [here](https://drive.google.com/drive/folders/1P9zbC_VMw_1K4DTTFFglLSN2J1PoI5kd?usp=sharing)) and MPI-INF-3DHP (from [here](https://drive.google.com/drive/folders/1yux7QiLOpHqJXVB9GaVz5A279JunGfuX?usp=sharing)). Put them in the `./checkpoint` directory. 

### Human3.6M

To evaluate our D3DP with JPMA using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/best_h36m_model -gpu 0 --evaluate best_epoch.bin -num_proposals 1 -sampling_timesteps 1 -b 4 --p2
```
to compare with the deterministic methods.  
Please run:
```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/best_h36m_model -gpu 0 --evaluate best_epoch.bin -num_proposals 20 -sampling_timesteps 10 -b 4 --p2
```
to compare with the probabilistic methods.  

You can balance efficiency and accuracy by adjusting `-num_proposals` (number of hypotheses) and `-sampling_timesteps` (number of iterations).

### MPI-INF-3DHP
To evaluate our D3DP with JPMA using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/best_3dhp_model -gpu 0 --evaluate best_epoch.bin -num_proposals 5 -sampling_timesteps 5 -b 4 --p2
```
After that, the predicted 3D poses under P-Best, P-Agg, J-Best, J-Agg settings are saved as four files (`.mat`) in `./checkpoint`. To get the MPJPE, AUC, PCK metrics, you can evaluate the predictions by running a Matlab script `./3dhp_test/test_util/mpii_test_predictions_ori_py.m` (you can change 'aggregation_mode' in line 29 to get results under different settings). Then, the evaluation results are saved in `./3dhp_test/test_util/mpii_3dhp_evaluation_sequencewise_ori_{setting name}_t{iteration index}.csv`. You can manually average the three metrics in these files over six sequences to get the final results. An example is shown in `./3dhp_test/test_util/H20_K10/mpii_3dhp_evaluation_sequencewise_ori_J_Best_t10.csv`.


## Training from scratch
### Human3.6M
To train our model using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/model_ddhpose_h36m -gpu 0 
```

### MPI-INF-3DHP
To train our model using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/model_ddhpose_3dhp -gpu 0 
```

## **Citation**
```bibtex
@inproceedings{cai2024disentangled,
  title={Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser},
  author={Cai, Qingyuan and Hu, Xuecai and Hou, Saihui and Yao, Li and Huang, Yongzhen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={882--890},
  year={2024}
}
```

## Acknowledgement
Our code refers to the following repositories.
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [D3DP](https://github.com/paTRICK-swk/D3DP)

We thank the authors for releasing their codes
