# AlzheimersDeepLearning

AlzheimersDeepLearning is an exploratory and educational project that delves into the application of deep learning for Alzheimer's disease research. This repository houses two main components:

- [MRIALZ](#alzheimers-disease-classification-using-mri-images-and-bayesian-hyperparameter-optimization): a convolutional neural network (CNN) for the classification of Alzheimer's disease stages using MRI scans
- [MRIALZ_DCGAN](#alzheimers-disease-mri-image-generation-using-dcgan): an exploration into leveraging AI to generate synthetic MRI images indistinguishable from real MRI images via Deep Convolutional Generative Adversarial Networks (DCGANs)

While the initial intention behind the DCGAN project was to augment MRI data for improved model training, I realized only once I had started generating decent images that this method could be problematic for training a model intended for real-life detection. Therefore, the DCGAN project serves more as a fun challenge to see how realistic the DCGAN MRI images could become.

**tldr;**
- [Very cool results from the final DCGAN architecture training (use the slider to watch noise transform into an MRI 'deep-fake')](https://wandb.ai/spindouken/mrialz_dcgan/reports/80-Epoch-Final-Run--Vmlldzo1ODU5MzU0?accessToken=lx3nmeo9e3l9l8crrbrf8y29pil5xrloi66l7tjlaerbnmj1zc1eqwwoxtby0szw)

## Supplemental Medium Articles

- [Optimizing Alzheimer's Disease Classification using Bayesian Optimization and Transfer Learning](https://medium.com/@masonthecount/optimizing-alzheimers-disease-classification-using-bayesian-optimization-and-transfer-learning-3f9ed8cbad56)
- [MRI Image Generation using Deep Convolutional GANs](https://medium.com/@masonthecount/mri-image-generation-using-deep-convolutional-gans-6a8ebbdcc57f)

## Dataset

The projects utilize the **Alzheimer MRI Dataset**, which contains MRI images specifically curated for Alzheimer's research. The dataset includes a total of **6400 MRI images**, resized to a uniform dimension of **128 x 128 pixels** and categorized into four distinct classes:

- **Class 1: Mild Demented** — 896 images
- **Class 2: Moderate Demented** — 64 images
- **Class 3: Non Demented** — 3200 images
- **Class 4: Very Mild Demented** — 2240 images

## Alzheimer's Disease Classification using MRI Images and Bayesian Hyperparameter Optimization

### Table of Contents

1. [About The Project](#about-the-project)
2. [Built With](#built-with)
3. [Key Concepts](#key-concepts)
4. [Roadmap](#roadmap)

### About The Project

This project focuses on building a Convolutional Neural Network (CNN) for classifying Alzheimer's disease using MRI scans.

The main goal of the project is to classify various stages of Alzheimer's disease using MRI images while optimizing the model using Bayesian Optimization.

#### Key Highlights:
- **Dataset**: [Alzheimer's MRI Dataset available on Kaggle](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset/)
- **Model Architecture**: CNN using MobileNetV2
- **Transfer Learning**: Leveraged pre-trained models for initial training
- **Bayesian Optimization**: Optimizing hyperparameters to improve model performance
- **Evaluation Metrics**: Accuracy, AUC-ROC, F1-Score, focusing on imbalanced data

### Built With
- **Python 3.8+**
- **TensorFlow 2.x** - Deep learning framework for building and training CNNs
- **GPyOpt** - Bayesian Optimization library
- **Keras** - High-level neural networks API
- **Matplotlib & Seaborn** - For visualizations and plotting results

### Key Concepts

Bayesian Optimization was used to fine-tune the hyperparameters, improving model performance without the computational burden of grid or random search.

Transfer learning was used by fine-tuning the MobileNetV2 model, which was pre-trained on ImageNet. Though pre-trained on a general dataset, transfer learning allows us to reduce training time while maintaining acceptable model accuracy.

F1-Score was used as a custom metric to handle imbalanced data, which is common in medical datasets like this. By focusing on both precision and recall, the F1-Score ensures better model performance across all classes.

#### Why Bayesian Optimization?
Traditional hyperparameter search methods, such as grid or random search, are inefficient and can be time-consuming. Bayesian optimization, however, uses a probabilistic model (in this case, a Gaussian Process) to predict the objective function and intelligently select the next set of hyperparameters to evaluate.

#### Gaussian Process Primer
A Gaussian Process (GP) models the objective function and estimates its uncertainty, enabling more informed exploration of hyperparameters. This is crucial in high-dimensional spaces where exhaustive search methods struggle.

#### Custom Metrics: F1-Score
The F1-Score is used as a custom metric to handle imbalanced data, which is common in medical datasets like this. By focusing on both precision and recall, the F1-Score ensures better model performance across all classes.

#### From-Scratch and Transfer Learning Methods
The project explored both from-scratch CNN models and transfer learning from pre-trained networks. This approach allowed me to compare the efficiency and accuracy of different training methodologies in classifying Alzheimer's stages.

### Roadmap
- Implement a CNN using MobileNetV2
- Integrate Bayesian Optimization for hyperparameter tuning
- Add F1-Score as a custom evaluation metric
- Explore different CNN architectures like ResNet or DenseNet
- Integrate additional MRI datasets for better model generalization

## Alzheimer's Disease MRI Image Generation using DCGAN

### Table of Contents

1. [About This Project](#about-this-project)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Baseline DCGAN Architecture](#baseline-dcgan-architecture)
   - [Generator](#generator)
   - [Discriminator](#discriminator)
4. [Experiment Variations](#experiment-variations)
5. [Built With](#built-with)
6. [Steps of Code](#steps-of-code)

### About This Project

This project focuses on using **Deep Convolutional Generative Adversarial Networks (DCGAN)** to generate synthetic MRI images of Alzheimer's disease. The primary goal is to understand GANs and their application in the medical imaging domain. By training the DCGAN, we aim to create high-quality MRI images from random noise, which can potentially augment existing datasets for better model training.

### Data Loading and Preprocessing

The `load_and_preprocess_mri` function uses **OpenCV** to handle image manipulation. The function:

- Reads each image in grayscale
- Resizes images to **128 x 128 pixels**
- Normalizes pixel values to the range of **[-1, 1]**

These preprocessed images are compiled into a NumPy array for training.

### Baseline DCGAN Architecture

#### Generator

The Generator network consists of several layers, including:

- **Dense Layer**: Takes a noise vector of size **100**
- **Batch Normalization**: Stabilizes activations
- **LeakyReLU Activation**: Introduces non-linearity
- **Reshape Layer**: Reshapes the tensor to **8x8x512**
- **Conv2DTranspose Layers**: Upscale the image and add finer details with each layer, mimicking the painting process where broad strokes evolve into intricate details
- **Output Layer**: Produces the final image with a single channel for grayscale

#### Discriminator

The Discriminator network includes:

- **Conv2D Layers**: Extract features from input images, with an increasing filter count for capturing complex patterns
- **LeakyReLU Activation**: Introduces non-linearity
- **Dropout Layer**: Reduces overfitting
- **Flatten Layer**: Flattens the tensor for the Dense layer
- **Dense Layer**: Outputs a probability indicating whether the input image is real or fake

### Experiment Variations

In initial experiments, **Residual Blocks** were added to the architecture. These blocks help mitigate the vanishing and exploding gradient problems often faced in deep networks. The use of residual connections allows for more efficient training of deep networks and has been shown to improve performance and convergence rates.


### Built With
Ensure you have the following libraries installed:
- **Python 3.8+**
- **TensorFlow 2.x** - Deep learning framework for building and training CNNs
- **opencv** - Library for working with dataframes
- **wandb** - Weights and Biases library

### Steps of code
1. **Data Loading**: Load the MRI images from the directory using ImageDataGenerator for training and validation.
2. **Model Creation**: Build the CNN based on MobileNetV2 architecture.
3. **Hyperparameter Optimization**: Tune the hyperparameters using GPyOpt to find the best combination.
4. **Final Training**: Train the model with the optimized hyperparameters and evaluate performance.

You can find all the code and run it in Google Colab by clicking the links in the repository.

## Creator
Mason Counts
