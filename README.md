

# GAVE

This is the official repository of the paper ["Dataset, Baseline and Evaluation Design for GAVE Challenge"](), which is the baseline of MICCAI2025 Challenge: ["Generalized Analysis of Vessels in Eye (GAVE)"](https://aistudio.baidu.com/competition/detail/1315)




## :star2: Highlights

* GAVE dataset provides pixel-level Artery/Vein  annotation of color fundus photographs . Through the innovative use of fluorescein fundus angiography (FFA) images paired with fundus color photos to assist the annotation process, vessel boundaries, small vessels and noise parts can be more accurately annotated.
* We also provide arteriovenous ratio(AVR) label of color fundus photographs. We are the first and biggest dataset to introduce FFA-assisted color fundus photo labeling and provide AVR labels.
* A novel recursive segmentation framework was proposed and as baseline of the GAVE dataset. We also provide detailed evalutation code regarding to classification and topology.



## :pushpin: Data format

![fig2](D:\typora_pic\fig2.PNG)

* Segmentation: Above is an example of our provided arteriovenous labels. Our code always expects the images to be RGB images with pixel values in the range [0, 255] and the labels to be RGB images with the following segmentation maps in each channel:

    + R: Artery
    + G: Interection of vessels
    + B: Vein

    The masks should be binary images with pixel values in the range [0, 255]. The predictions will be saved in the same format as the masks. To align with the model prediction, in training stage and evaluation stage, our code uses the union of R and G channels (intersection) as the artery label, the union of B and G channels (intersection) as the vein label, and the union of R,G,B channels as the vessel label. This process just for label, not for prediction. It can be found in the related code. Our training output is three channels:
    
	+ R: Artery
    + G: Vessel
    + B: Vein

Here's an example of a model prediction, its RGB channels represents artery, vessel and vein respectively.

![fig_example](D:\typora_pic\fig_example.png)



## :sunny: Preparation


The code was tested using Python3.10.12.
However, it should work with other Python versions and package managers.
Just make sure to install the required packages listed in `requirements.txt`. 


### :hammer: Environment settings

> [!IMPORTANT]
> Make sure you have installed conda in advance.

Create and activate Python environment
```
conda create -n gave python==3.10
conda activate gave
```

Update `pip`.

```sh
pip install --upgrade pip
```

Install requirements using `requirements.txt`.

```sh
pip3 install -r requirements.txt
```

```sh
sudo dnf install clang
```

Install Python version 3.10.10.
```sh
CC=clang pyenv install -v 3.10.10
```

### :wrench: Preparing Dataset

You can download the GAVE dataset through the ["GAVE challenge"](https://aistudio.baidu.com/competition/detail/1315) on AI studio. Put the dataset in the `./Data`. The dataset directory structure is following:
```sh
|-Data
	|-GAVE
		|-training
			|-av   		# artery/vein label
			|-images	# color fundus images
			|-masks		# ROI masks
```

### :telescope: Project structure

```sh
|-Code
|	|-Config		# for training
|	|...
|	|-AVR.ipynb 	# AVR measurement temple code
| 	|-get_pred.py 	# step 2
|	|-test.py		# step 3
|	|-train.py		# step 1
|-Data				# Dataset in here
|	|-GAVE
|-Log
	|-log1			# Your training/test result, weight
```

### :anchor: Preprocess (optional)

You can preprocess the images offline using the `preprocess.py` script which in the directory `Code/Tool/`. The script will enhance the images and masks and save them in the specified directory.

```bash
python preprocess.py --i data/images/ --m data/masks/ --s data/enhanced
```



## :rocket: Run your code

### :one: Training

All training code can be found through the entrance of training script `train.py`, and the configuration file, with all the hyperparameters and command line arguments, is `cfg.py`.

```bash
python train.py --dataset GAVE --model RRWNet --version trainv0
```
Training logs, model.py, learning curve and best/latest weights will be saved under the `Log/ directory`. You can use `--version` parameter to specify the training id  for your train directory name. More parameter pls refer to `cfg.py`.


### :two: Get predictions

After the model trained, the predictions can be generated using the following command. If you use preprocess for training dataset, pls do the same for test.

```bash
python get_pred.py -p <path_to_the_trained_model> -i <path_to_the_images> -m <path_to_the_ROI_masks> -t <specify_result_id>
```

The predictions will be saved under the `pred/test-name` directory in the path specified by the `-p` flag and `-t`flag.


### :three: Evaluation

You can evalutate the predictions with Groundtruth using `test.py`, after your training and got the result. Or you can directly evaluate any predictions with gt,  just make sure there paired. The evalutation result will be saved to json file under `--results_dir` which you can specify it. You can specify the  json filename using `-d`.  We provide metric covering multi-dimension, including Sens, Spec, Acc, F1, AUC, DICE, clDice, HD95, Topology(INF/COR).

```bash
python test.py -d <specify_eval_id> -p <path_to_the_predictions> -g <path_to_the_ground_truth> -m <path_to_the_ROI_masks> -s <img_size_heightxwidth>
```

You can always run `python test.py --help` to see the available options.




##  :microscope:AVR measurement
Following the clinical experience of ophthalmologists, we used the arteriovenous diameter top4 at the optic disc edge to calculate AVR. Specifically, in Baseline, we used the official pre-trained MNet to segment the optic disc and extract the contour, and then overlapped with the previously segmented arteriovenous vessels to extract the arteriovenous diameters of the optic disc edge and took the first four thickest diameters to calculate AVR. The implementation of MNet is available on ["MNet_DeepCDR"](https://github.com/HzFu/MNet_DeepCDR) and will not be described here.
We provide an example code "AVR.ipynb " for AVR measurement for you reference. 




## :email: Contact

If you have any questions or problems with the code or the paper, please do not hesitate to open an issue in this repository (preferred) or contact me at `liu_zw0@163.com`.


## :thumbsup: Acknowledge
Our project code is built based on the [rrwnet](https://github.com/j-morano/rrwnet) project. The authors' outstanding work, code  and  kind help are gratefully acknowledged.

