
# **Food Image Classification with Domain Adaptation**

A deep learning pipeline for training food classifiers using synthetic (GAN-generated) images and adapting them to real images using **Domain-Adversarial Neural Networks (DANN)**.

## **Overview**

This repository implements a complete workflow for food image classification across three categories: **pizza**, **sushi**, and **pasta**.  
The project addresses the **domain gap** between synthetic and real images through adversarial domain adaptation.


## **Key Features**

- **Automated Data Collection**
  - Scrape and download food images using the Pexels API
- **Advanced Data Augmentation**
  - Supports multiple augmentation strategies (Albumentations, DiffAugment)
- **Domain Adaptation**
  - Bridge the synthetic → real gap using **DANN**
- **Class Imbalance Handling**
  - Weighted sampling and weighted loss functions
- **Comprehensive Evaluation**
  - t-SNE visualization, confusion matrices, and per-class metrics

## **Requirements**

Install dependencies:

```bash
pip install torch torchvision
pip install albumentations
pip install imagehash
pip install scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install requests pillow
```
## **Usage**

- **Data Collection**: Download food images from Pexels API
  - Features:
      - Parallel downloading with thread pools
      - Duplicate detection using perceptual hashing (pHash)
      - Automatic image resizing and validation
      - Min resolution filtering 
- **Data Augmentation**
  - Load and augment datasets for training
  - **Available Augmentation Methods**:
      - DiffAugment (for GAN training):
        - Color adjustments (brightness, saturation, contrast)
        - Random translation
        - Cutout regularization
      - Albumentations (for diffusion models):
        - Horizontal flips
        - Random crops
        - Color jittering
        - Gaussian blur/noise
        - Affine transformations
        - Shift-scale-rotate
- **Domain Adaptation**
  - Train a classifier on synthetic images and adapt it to real images
  - Training Pipeline:
    - Source Training: Train ResNet-18 classifier on GAN images
    - Domain Adaptation: Apply DANN to learn domain-invariant features
    - Evaluation: Test on real images and visualize feature spaces

## **Methodology**
  Domain-Adversarial Neural Networks (DANN) architecture consists of three components
  
  - Feature Extractor: ResNet-18 backbone (pre-trained on ImageNet)
  - Class Classifier: Predicts food category (pasta/pizza/sushi)
  - Domain Classifier: Distinguishes between GAN and real images
  
  The Gradient Reversal Layer (GRL) creates adversarial training where
  
  - Feature extractor learns to fool the domain classifier
  - Domain classifier tries to distinguish source from target
  - Result: Domain-invariant features that work on both GAN and real images

## **Class Imbalance Handling**
Two strategies are implemented:
  - Weighted Loss: CrossEntropyLoss with class weights inversely proportional to frequency
  - Balanced Sampling: WeightedRandomSampler ensures equal probability across classes

## **Visualizations**
The pipeline generates:
  - t-SNE Feature Space: Before/after adaptation comparison
  - Training Metrics: Classification loss and domain discriminator accuracy
  - Confusion Matrix: Per-class performance breakdown

## **Key Parameters**
**Data Collection**
  - target: Images per category (default: 300)
  - img_size: Resolution for downloaded images (default: 1024)
  - threads: Parallel download workers (default: 10)

**Training**
  - EPOCHS_SOURCE: Source model training epochs (default: 10)
  - EPOCHS_ADAPT: Domain adaptation epochs (default: 15)
  - BATCH_SIZE: Training batch size (default: 32)
  - train_split: Train/test split ratio (default: 0.8)

**Augmentation**
  - augment_prob: Probability of applying augmentations (default: 0.8)
  - policy: DiffAugment operations (default: 'color,translation,cutout')

## **Notes**

- Pexels API: Requires a free API key from Pexels
- Device Support: Auto-detects CUDA, MPS (Apple Silicon), or CPU
- Memory Requirements: Adjust batch size based on available GPU memory
- Pretrained Weights: Uses ImageNet-pretrained ResNet-18 for feature extraction
