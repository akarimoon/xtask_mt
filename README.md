# Cross-Task Self-Consistency for Multi-Task Learning

## Tested on
- numpy(v1.19.1)
- opencv-python(v4.4.0.42)
- torch(v1.7.0)
- torchvision(v0.8.0)
- tqdm(v4.48.2)
- matplotlib(v3.3.1)
- seaborn(v0.11.0)
- pandas(v.1.1.2)

## Data
### Cityscapes (CS)
Download [`Cityscapes` dataset](https://www.cityscapes-dataset.com/login/) and put it in a subdirectory named `./data/cityscapes`.
The folder should have the following subfolders:
- RGB image in folder `leftImg8bit`
- Segmentation in folder `gtFine`
- Disparity maps in folder `disparity`

### NYU
We use the preprocessed [`NYUv2` dataset](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) provided by [this repo](https://github.com/lorenmt/mtan). Download the dataset and put it in the dataset folder in `./data/nyu`.

## Model
The model consists of one encoder (ResNet) and two decoders, one for each task. 
The decoders outputs the predictions for each task ("direct predictions"), which are fed to the _TaskTransferNet_.<br>
The objective of the TaskTranferNet is to predict the other task given a prediction image as an input 
(Segmentation prediction -> Depth prediction, vice versa), which I refer to as "transferred predictions"

## Loss function
When computing the losses, the direct predictions are compared with the target 
while the transferred predictions are compared with the direct predictions so that they "align themselves".<br>
The total loss consists of 4 different losses:
- direct segmentation loss: _CrossEntropyLoss()_
- direct depth loss: _L1()_ or _MSE()_ or _logL1()_ or _SmoothL1()_
- transferred segmentation loss: <br>_CrossEntropyLoss()_ or _KLDivergence()_
- transferred depth loss: _L1()_ or _SSIM()_

\* Label smoothing: To "smooth" the one-hot probability by taking some of the probability from the correct class and distributing it among other classes.<br>
\* SSIM: [Structural Similarity Loss](https://github.com/Po-Hsun-Su/pytorch-ssim)

## Flags
The flags are the same for both datasets. The flags and its usage are as written below,

| Flag Name        | Usage  |  Comments |
| ------------- |-------------| -----|
| `input_path`     | Path to dataset  | default is `data/cityscapes` (CS) or `data/nyu` (NYU)|
| `height`   | height of prediction | default: 128 (CS) or 288 (NYU) |
| `width`   | width of prediction | default: 256 (CS) or 384 (NYU) |
| `epochs`   | # of epochs | default: 250 (CS) or 100 (NYU) |
| `enc_layers`   | which encoder to use | default: 34, can choose from 18, 34, 50, 101, 152 |
| `use_pretrain`   | toggle on to use pretrained encoder weights | available for both datasets |
| `batch_size`   | batch size | default: 6 |
| `scheduler_step_size`   | step size for scheduler | default: 80 (CS) or 60 (NYU), note that we use StepLR |
| `scheduler_gamma`   | decay rate of scheduler | default: 0.5 |
| `alpha`   | weight of adding transferred depth loss | default: 0.01 (CS) or 0.0001 (NYU) |
| `gamma`   | weight of adding transferred segmentation loss | default: 0.01 (CS) or 0.0001 (NYU) |
| `label_smoothing`   | amount of label smoothing | default: 0.0 |
| `lp`   | loss fn for direct depth loss | default: L1, can choose from L1, MSE, logL1, smoothL1 |
| `tdep_loss`   | loss fn for transferred depth loss | default: L1, can choose from L1 or SSIM |
| `tseg_loss`   | loss fn for transferred segmentation loss | default: cross, can choose from cross or kl |
| `batch_norm`   | toggle to enable batch normalization layer in TaskTransferNet | slightly improves segmentation task |
| `wider_ttnet`   | toggle to double the # of channels in TaskTransferNet |  |
| `uncertainty_weights`   | toggle to use uncertainty weights (Kendall, et al. 2018) | we used this for best results |
| `gradnorm`   | toggle to use GradNorm (Chen, et al. 2018) |  |

## Training
### Cityscapes
For the Cityscapes dataset, there are two versions of segmentation task, which are 7-classes task and 19-classes task (Use flag 'num_classes' to switch tasks, default is 7).<br>
So far, the results show near-SOTA for 7-class segmentation task + depth estimation.

ResNet34 was used as the encoder, _L1()_ for direct depth loss and _CrossEntropyLoss()_ for transferred segmentation loss.<br>
The hyperparameter weights for both transferred predictions were 0.01.<br>
I used Adam as my optimizer with an initial learning rate of 0.0001 and trained for 250 epochs with batch size 8. The learning rate was halved every 80 epochs.

To reproduce the code, use the following:
```
python main_cross_cs.py --uncertainty_weights
```

### NYU
Our results show SOTA for NYU dataset.

ResNet34 was used as the encoder, _L1()_ for direct depth loss and _CrossEntropyLoss()_ for transferred segmentation loss.<br>
The hyperparameter weights for both transferred predictions were 0.0001.<br>
I used Adam as my optimizer with an initial learning rate of 0.0001 and trained for 100 epochs with batch size 6. The learning rate was halved every 60 epochs.

To reproduce the code, use the following:
```
python main_cross_nyu.py --uncertainty_weights
```

## Comparisons
Evaluation metrics are the following:

<ins>Segmentation</ins>
- Pixel accuracy (Pix Acc): percentage of pixels with the correct label
- mIoU: mean Intersection over Union

<ins>Depth</ins>
- Absolute Error (Abs)
- Absolute Relative Error (Abs Rel): Absolute error divided by ground truth depth

The results are the following:

### Cityscapes
| Models                | mIoU  |Pix Acc|Abs     |Abs Rel |
|:---------------------:|:-----:|:-----:|:------:|:------:|
| MTAN                  | 53.04 | 91.11 | 0.0144 | 33.63  |
| KD4MTL                | 52.71 | 91.54 | 0.0139 | 27.33  |
| PCGrad                | 53.59 | 91.45 | 0.0171 | 31.34  |
| AdaMT-Net             | <ins>62.53</ins> | **94.16** | <ins>0.0125</ins> | <ins>22.23</ins>  |
| Ours                  | **66.51**	| <ins>93.56</ins> | **0.0122** | **19.40**  |
### NYU
| Models     |mIoU   |Pix Acc| Abs    |Abs Rel |
|:----------:|:-----:|:-----:|:------:|:------:|
| _MTAN\*_   | 21.07 | 55.70 | 0.6035 | 0.2472 |
| _MTAN†_    | 20.10 | 53.73 | 0.6417 | 0.2758 |
| KD4MTL\*   | 20.75 | 57.90 | 0.5816 | 0.2445 |
| _KD4MTL†_  | <ins>22.44</ins> | 57.32 | <ins>0.6003</ins> | 0.2601 |
| PCGrad\*   | 20.17 | 56.65 | 0.5904 | 0.2467 |
| _PCGrad†_  | 21.29 | 54.07 | 0.6705 | 0.3000 |
| AdaMT-Net\*| 21.86 | 60.35 | 0.5933	| 0.2456 |
| AdaMT-Net† | 20.61 | <ins>58.91</ins> | 0.6136 | <ins>0.2547</ins> |
| Ours†      | **30.31** | **63.02** | **0.5954** | **0.2235** |

\*: Trained on 3 tasks (segmentation, depth, and surface normal)<br>
†: Trained on 2 tasks (segmentation and depth)<br>

Scores with models trained on 3 tasks for NYU dataset are shown only as reference.

### Papers referred
MTAN: \[[paper](https://arxiv.org/pdf/1803.10704.pdf)\]\[[github](https://github.com/lorenmt/mtan)\]<br>
KD4MTL: \[[paper](https://arxiv.org/pdf/2007.06889.pdf)\]\[[github](https://github.com/VICO-UoE/KD4MTL)\]<br>
PCGrad: \[[paper](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)\]\[[github (tensorflow)](https://github.com/tianheyu927/PCGrad)\]\[[github (pytorch)](https://github.com/WeiChengTseng/Pytorch-PCGrad)\]<br>
AdaMT-Net: \[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Jha_AdaMT-Net_An_Adaptive_Weight_Learning_Based_Multi-Task_Learning_Model_for_CVPRW_2020_paper.pdf)\]


