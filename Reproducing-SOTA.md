### Single-Instance Learning Models
The following details were used for the reproducibility experiments of image-level prediction on the official CBIS split (Table II).

1. Kim et al. [1] <br/>
The details are specific to the following models: DIB-MG <br/>
**Training:** We implemented the DIB-MG model using the information reported in the paper. Specifially, we use a learning rate decay scheduler starting from lr=0.001 and decaying by a factor of 0.2 every 10 epochs. Weight decay was set to 0.0005. We used SGD optimizer with momentum 0.9 for training. Models were trained till 50th epoch with early stopping (patience epoch 10 on validation loss) on validation set. DIB-MG uses 2 output neurons (one for benign, one for malignant) with softmax. Cross-entropy loss was used for training the model.  We repeated the experiments with 3 random seeds (8, 24, 80) for weight initialization and reported the mean and standard deviation across these 3 runs. <br/>
**Data Augmentation:** We randomly perturbed the brightness and contrast by 10%, resized to 1600x1600 and normalized the images to the range of [-1,1]. Before resizing, we zero-padded the shorter side of the image to 1600 (left side for RCC/RMLO and right side for LCC/LMLO). The images were not flipped horizontally to retain their original orientation.

2. Shu et al. [2] <br/>
The details are specific to the following models: DenseNet169 (+avgpool, +maxpool, +RGP, +GGP) <br/>
**Training:** We use a learning rate decay scheduler starting with lr=0.00002 for CNN layers and lr=0.0001 for classification layer and decayed the lr every 8 epochs. We used Adam optimizer for training. No validation set was used for early stopping. The model was trained till 150th epoch and the performance on the test set at this epoch was reported. The models use one output neuron with sigmoid activation. Binary cross-entropy loss was used for training the model. We repeated the experiments with 3 random seeds (8, 24, 80) for weight initialization and reported the mean and standard deviation across these 3 runs.  <br/>
**Data Augmentation:** We resized the images to 800x800, followed by normalization within the range [0,1] by dividing by 255. Then, the images were randomly flipped horizontally with a probability of 0.5, contrast and saturation were randomly perturbed by 20%, followed by random rotation of 30 degrees and addition of gaussian noise with mean 0 and standard deviation 0.005.

3. Shen et al. [3] <br/>
The details are specific to the following models: ResNet34, GMIC-ResNet18 <br/>
**Training:** We ran 20 different hyperparameter configuration on learning rate, weight decay and regularization term $\beta$ (applied on their sparsity term in loss function). We searched for a learning rate within $10^{[-4,-5.5]}$, weight decay within $10^{[-3.5,-5.5]}$, $\beta$ within $10^{[-3.5,-5.5]}$ (similar to [3]). Models were trained till 50th epoch and the model at the epoch with best validation AUC was selected as the final model. We trained the models with custom loss function used in [3]. We set the $t$ in top t% pooling to 0.02 and set the number of ROI patches to 6. For GMIC-ResNet18, we used a separate pretrained ResNet18 feature instantiation for both, the local and the global network. We took the 5 best hyperparameter combinations based on the validation AUC and reported their mean and std dev in reproducibility experiment. <br/>
**Data Augmentation:** Images were resized to 2944x1920. We randomly flipped the images horizontally with probability 0.5, applied random affine transformation consisting of rotation 15 degrees, translation upto 10% of the image size, scaling by a random factor between 0.8 and 1.6, random shearing 25 degrees and added gaussian noise of mean 0 and std dev 0.005.

### Multi-Instance Learning Models
We used official split for VinDr, but created our own splits for CBIS and MGM such that all cases of a patient were exclusively contained in a subset. We train the models on CBIS and VinDr for a maximum of 50 epochs and MGM for 30 (due to a longer training time for MGM). We repeat the experiments 2 times for MGM (random seeds 8 and 24) and 3 times for CBIS (random seeds 8, 24, 42) with different random seeds on dataset split. For VinDr, we repeat the experiments with 3 random seeds (8, 24, 80) on weight initialization and reported the mean and standard deviation across these runs. <br>

1. Kim et al. [1] <br/>
All training and data augmentation details are the same as those mentioned above for Kim et al. under Single-instance Learning models. 

2. Wu et al. [4] <br/>
**Training:** As reported in their paper, we used a learning rate of $10^{-5}$, weight decay of $10^{-4.5}$ and used Adam optimizer for training. We stopped the model training if the validation AUC did not improve for 20 epochs. We resize the images to 2944x1920 for MGM and CBIS following the original paper. However, VinDr was resized to 2700x990, as the original image size was smaller than the resize dimension used in GMIC (specifically the width). So, we resized the images to mean and std dev of height and width of all images in the dataset. The models use one output neuron with sigmoid activation. Binary cross-entropy loss was used for training the model. <br/>
**Data Augmentation:** We followed the data augmentation of Shen et al. (GMIC-ResNet18) described above under Single-instance Learning models.  

3. Our MIL variants <br/>
**Training:** We took the best hyperparameter combination from the GMIC hyperparameter tuning on CBIS for image-level models and used it for MIL experiments with GMIC as a feature extractor for all datasets. All models were trained with Adam optimizer and the training was stopped if the validation AUC did not improve for 10 epochs. We resize the images to 2944x1920 for MGM and CBIS following the original paper. However, VinDr was resized to 2700x990, as the original image size was smaller than the resize dimension used in GMIC (specifically the width). So, we resized the images to mean and std dev of height and width of all images in the dataset. <br/>
The best hyperparameter combination used in our experiments are: <br/>
lr $= 0.0000630957344480193$ <br/>
wtdecay $= 0.000316227766016838$ <br/>
regularization term $\beta = 0.000158489319246111$ <br/>
**Data Augmentation:** We followed the data augmentation of Shen et al. (GMIC-ResNet18) described above under Single-instance Learning models. 

### General Settings
For the val and test set, the images were only resized and normalized. The single channel of grayscale images was duplicated into 3 channels to fit pretrained models and normalized to ImageNet mean and standard deviation. For training from-scratch models, only the single grayscale channel was used as input and normalized to the range of [-1,1] [1].

### References

1. E.-K. Kim, H.-E. Kim, K. Han, B. J. Kang, Y.-M. Sohn, O. H. Woo, and C. W. Lee, “Applying data-driven imaging biomarker in mammography for breast cancer screening: preliminary study,” Scientific reports, vol. 8, no. 1, pp. 1–8, 2018. <br/>
2. X. Shu, L. Zhang, Z. Wang, Q. Lv, and Z. Yi, “Deep neural networks with region-based pooling structures for mammographic image classification,” IEEE transactions on medical imaging, vol. 39, no. 6, pp. 2246–2255, 2020. <br/>
3. Y. Shen, N. Wu, J. Phang, J. Park, K. Liu, S. Tyagi, L. Heacock, S. G. Kim, L. Moy, K. Cho et al., “An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization,” Medical image analysis, vol. 68, p. 101908, 2021. <br/>
4. N. Wu, J. Phang, J. Park, Y. Shen, Z. Huang, M. Zorin, S. Jastrzebski, T. Févry, J. Katsnelson, E. Kim, S. Wolfson, U. Parikh, S. Gaddam, L. L. Y. Lin, K. Ho, J. D. Weinstein, B. Reig, Y. Gao, H. Toth, K. Pysarenko, A. Lewin, J. Lee, K. Airola, E. Mema, S. Chung, E. Hwang, N. Samreen, S. G. Kim, L. Heacock, L. Moy, K. Cho, and K. J. Geras, “Deep neural networks improve radiologists’ performance in breast cancer screening,” IEEE Transactions on Medical Imaging, vol. 39, no. 4, p. 1184–1194, Apr 2020.
