#### Single-Instance Learning Models
The following details were used for the reproducibility experiments in SIL setting on the official cbisddsm split. For experiments where random seed was used to create a dataset split, we fixed the weight initialization random seed to 8. 

DIB-MG
We use a learning rate decay scheduler starting from lr=0.001 and decaying by a factor of 0.2 every 10 epochs. Weight decay was set to 0.0005. We used SGD optimizer with momentum 0.9 was training. Models were trained till 50th epoch with early stopping (patience epoch 10 on validation loss) on validation set. Experiments were repeated with 3 random seed for weight initialization. Cross-entropy loss was used for training the model.

Shu et al:
We use a learning rate decay scheduler starting with lr=0.00002 for CNN layers and lr=0.0001 for classification layer and decayed the lr every 8 epochs. We used Adam optimizer for training. No validation set was used for early stopping. The model was trained till 150th epoch and the performance on the test set at this epoch was reported. Experiments were repeated with 3 random seed for weight initialization. Binary cross-entropy loss was used for training the model.

GMIC:
we ran 20 different hyperparameter configuration on learning rate, weight decay and regularization term $\beta$ (applied on their sparsity term in loss function). We searched for a learning rate within $10^{[-4,-5.5]}$, weight decay within $10^{[-3.5,-5.5]}$, $\beta$ within $10^{[-3.5,-5.5]}$ (similar to~\cite{shen2021interpretable}).  We took the 5 best hyperparameter combinations based on the validation AUC and reported their mean and std dev in reproducibility experiment. Models were trained till 50th epoch and the model checkpointed at the epoch with best validation AUC. We trained the models with custom loss function used in~\cite{shen2021interpretable}. We set the $t$ in top t\% pooling to 0.02 for all our experiments including GMIC and set the number of ROI patches to 6. 
