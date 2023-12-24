# FCBA
This repo is implementation for the accepted paper "Beyond Traditional Threats: A Persistent Backdoor Attack in Federated Learning" ( AAAI 2024 )

![avatar](https://github.com/PhD-TaoLiu/FCBA/blob/main/FCBA-visio-show.jpg)

Figure 1: Overview of full combination backdoor attack(FCBA) in FL. At round *t* + 1, the aggregator merges local data (both benign and adversarial) from *t* to update G<sub>*t*+1</sub>.During a backdoor attack, the attacker uses trigger partition *m* to create local trigger patterns and identifies *M* maliciousclients, each with a unique trigger pattern.

### Requirements

```
Python >= 3.7
Pytorch >= 1.12.1
numpy >= 1.21.5
```

We conducted experiments using two parallel NVIDIA GeForce RTX 3090 graphics cards.

### Datasets

CIFAR10 will be automatically download

### Folder Structure

```
.
├── data                        # Data Storage
├── models                      # Supported models
├── saved_models				# Save experimental data
├── utils						# Experimental configuration folder
├── helper.py                   # a utility module designed to assist in the              									training and communication of deep learning models.
├── image_helper.py             # A helper class for processing image data, mainly 										  used for loading, processing, and preparing data 
								  in deep learning models.
├── image_train.py              # Defined a function for training deep learning 									 	  models
├── main.py                     # Federated framework main function
├── test.py                     # Test class functions to test model performance
├── train.py					# Execute model training based on input parameters 										  and configuration information
```

### How to run: 

- prepare the pretrained model:
  Our clean models for CIFAR10 have been placed in the directory \saved_models\cifar_pretrain. You can also train from the round 0 to obtain the pretrained clean model.
- we can use Visdom to monitor the training progress.

```
python -m visdom.server -p 8096
```

- set parameters properly,  then run experiments for the datasets:

```
python main.py --params utils/cifar.yaml
```

### Optional Parameters

There are a several of optional arguments in the `cifar_params.yaml`:

- `lr`/`poison_lr`: Client/Malicious client learning rate

- `internal_poison_epochs` : Benign client training rounds per round

- `internal_poison_clean_epochs` : Malicious clients have training rounds per round.

- `number_of_total_participants` : Number of participating clients.

- `scale_weights_poison` : Malicious client amplifies parameters.

- `total_list` : List of Malicious Client Numbers.

- `X_poison_pattern` : Add position of the Xth trigger

- `total_list` : List of Malicious Client Numbers.

- `X_poison_pattern` : Add position of the Xth trigger

- `X_poison_epochs` : The X-th type trigger malicious client adds a round

- `dirichlet_alpha` : If you choose to partition the client using the Dirichlet distribution, this is the coefficient
