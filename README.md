# ActSemiCNNAct
This README would normally document whatever steps are necessary to get your application up and running.

Installation Clone the code and its submodules

git clone git@bitbucket.org:irc-sphere/semisupervision_attn_cnn.git
cd ActSemi_CNN_Act
git submodule update --init --recursive
This code is working with Python3.6.

python3.6 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
Introduction to the folders and files in this repo:

architectures: the backbone network strcutures can be used
data-local: the datasets which can be used for validation
trainer: temporal ensembling based model training
ActSemi_PAMAP_Test06.py: mail file for training and testing with PAMAP participant 06 data as testset
ActSemi_PAMAP_LOSO.py: Leave one subject out cross validation on PAMAP dataset
Please run below lines to implement the method.

python main_USCHAD_LOSO.py
python ActSemi_PAMAP_Test06.py The results will be saved to 'results' file in the root path of the repo, which will be automatically created if not existing when running the experiments.
The input files should be put at the '/data-local' directly in the format of .npy with size Ns100Nf, where Ns is the number of samples, 100 means there are 100 time points in one sample, Nf is the nubmer of features.

This method is designed to tackle the limited label issue. The main ideas of the method are as below:

Combine active learning and semisupervised learning to select informative samples and make use of unlabelle data
Apply a temporal ensembling-based semisupervised approach to achieve a consensus prediction using the outputs of the training networks on different epochs under network dropout regularization. (An ensemble of multiple networks generally generates more robust results than a single network)
