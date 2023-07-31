### Single-Instance Learning Models
The following details were used for the reproducibility experiments of image-level prediction on the official CBIS split (Table II).

DIB-MG <br/>
**Training:** We implemented the DIB-MG model using the information reported in the paper. Specifially, we use a learning rate decay scheduler starting from lr=0.001 and decaying by a factor of 0.2 every 10 epochs. Weight decay was set to 0.0005. We used SGD optimizer with momentum 0.9 for training. Models were trained till 50th epoch with early stopping (patience epoch 10 on validation loss) on validation set. Cross-entropy loss was used for training the model. We repeated the experiments with 3 random seeds (8, 24, 80) for weight initialization and reported the mean and standard deviation across these 3 runs. <br/>
**Data Augmentation**: We randomly perturbed the brightness and contrast by 10%, resized to 1600x1600 and normalized the images to the range of [-1,1]. Before resizing, we zero-padded the shorter side of the image to 1600 (left side for RCC/RMLO and right side for LCC/LMLO). The images were not flipped horizontally to retain their original orientation.

Shu et al <br/>
**Training:** We use a learning rate decay scheduler starting with lr=0.00002 for CNN layers and lr=0.0001 for classification layer and decayed the lr every 8 epochs. We used Adam optimizer for training. No validation set was used for early stopping. The model was trained till 150th epoch and the performance on the test set at this epoch was reported. Experiments were repeated with 3 random seed for weight initialization. Binary cross-entropy loss was used for training the model. <br/>
**Data Augmentation**:We resized the images to 800x800, followed by normalization within the range [0,1] by dividing by 255. Then, the images were randomly flipped horizontally with a probability of 0.5, contrast and saturation were randomly perturbed by 20\%, followed by random rotation of 30 degrees and addition of gaussian noise with mean 0 and standard deviation 0.005.

GMIC <br/>
we ran 20 different hyperparameter configuration on learning rate, weight decay and regularization term $\beta$ (applied on their sparsity term in loss function). We searched for a learning rate within $10^{[-4,-5.5]}$, weight decay within $10^{[-3.5,-5.5]}$, $\beta$ within $10^{[-3.5,-5.5]}$ (similar to~\cite{shen2021interpretable}).  We took the 5 best hyperparameter combinations based on the validation AUC and reported their mean and std dev in reproducibility experiment. Models were trained till 50th epoch and the model checkpointed at the epoch with best validation AUC. We trained the models with custom loss function used in~\cite{shen2021interpretable}. We set the $t$ in top t\% pooling to 0.02 for all our experiments including GMIC and set the number of ROI patches to 6. 

#### Multi-Instance Learning Models

DIB-MG <br/>


DMV-CNN <br/>

