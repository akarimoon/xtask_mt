# Cross-task Alignment Learning for Multiple Tasks

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
Download Cityscapes dataset and put it in a subdirectory named `./data/cityscapes`.
The folder should have the following subfolders:
- RGB image in folder `leftImg8bit`
- Segmentation in folder `gtFine`
- Disparity maps in folder `disparity`

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
- direct depth loss: _L1()_ or _MSE()_ or _logL1()_
- transferred segmentation loss: <br>_CrossEntropyLosswithLabelSmoothing()_ or _KLDivergence()_
- transferred depth loss: _SSIM()_

\* Label smoothing: To "smooth" the one-hot probability by taking some of the probability from the correct class and distributing it among other classes.
\* SSIM: Structural Similarity Loss: https://github.com/Po-Hsun-Su/pytorch-ssim

When combining these losses, the naive method is to use a hyperparameter and take the weighted average. 
However, there is another method called "uncertainty weights" which is a learnable weights to balance the losses (Kendall, et al. 2018).
In this experiment, I used hyperparameters to balance the losses of the same task and uncertainty weights to balance the losses of the two tasks.

## Training
For the CityScapes dataset, there are two versions of segmentation task, which are 7-classes ver. and 19-classes ver.<br>
So far, the results show SOTA(?) in the 7-classes ver.

ResNet34 was used as the encoder, _logL1()_ for direct depth loss and _CrossEntropyLosswithLabelSmoothing()_ with smoothing value of 0.1 for transferred segmentation loss.<br>
The hyperparameter weight for depth loss was 0.4 and 0.001 for segmentation loss.<br>
I used Adam as my optimizer with a learning rate of 0.0001 and trained for 50 epochs. The learning rate was multiplied by 0.1 every 15 epochs.

To reproduce the code, use the following:
```
python main_cross.py -n 7 --alpha 0.4 --gamma 0.0001 --label_smoothing 0.1 --uncertainty_weights --lp logL1
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
| Models                |Pix acc|mIoU   |Abs     |Abs Rel |
|:---------------------:|:-----:|:- ---:|:------:|:------:|
| MTAN                  | 53.86 | 91.11 | 0.0144 | 33.63  |
| Knowledge Distillation| 52.71 | 91.54 | 0.0139 | 27.33  |
| Ours                  | 57.80 | 93.00 | 0.0169 | 0.6344 |

MTAN: https://arxiv.org/pdf/1803.10704.pdf
Knowledge Distillation: https://arxiv.org/pdf/2007.06889.pdf

## Some issues
- Codes for mIoU and IoU are from MTAN's github page. However, there IoU score does not match with the IoU score I've been using which is from a different git repo.
- The images used for both works have been "preprocessed" in some way, so there may be differences
- I have not found out why Absolute Relative Error is this low


