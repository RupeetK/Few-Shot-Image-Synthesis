

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
import os
from torch.utils.data import random_split
from collections import Counter





device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device {device}')





# Handling class imbalance

def compute_class_weights(dataset):
    '''
    Compute class weights for imbalanced datasets
    Weight = 1 / (class frequency)
    '''

    #Get all labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [label for _, label in dataset]

    #Count samples per class
    class_weights = []
    total_samples = len(labels)
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    unique_labels = sorted(class_counts.keys())

    for i in unique_labels:
        weight = total_samples / (num_classes * class_counts[i])
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights)

    print(f'\nClass Distribution: {dict(class_counts)}')
    print(f'Class weights: {class_weights.numpy()}')

    return class_weights

def create_balanced_sampler(dataset):
    '''
    Create WeightedRandomSampler for balanced batch sampling
    Ensures each class has equal probability of being sampled
    '''

    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [label for _, label in dataset]

    class_counts = Counter(labels)

    #Compute sample weights
    sample_weights = []

    for label in labels:
        #Weight inversely proportional to class frequency
        sample_weights.append(1.0 / class_counts[label])

    sample_weights = torch.DoubleTensor(sample_weights)

    # Create sampler
    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    return sampler









#DATA PREPARATION

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

def load_datasets(gan_path, real_path, batch_size = 32, num_workers = 2,train_split = 0.8, use_balanced_sampling = True):
    '''
    Load GAN-generated and real image datasets with class balancing and train/test splits 

    Args:
        gan_path: Path to GAN-generated images folder
        real_path: Path to real images folder

    Returns: gan_train_loader, gan_test_loader, real_train_loader, real_test_loader,
        num_classes, class_names
    '''
    # Load datasets 

    gan_dataset = ImageFolder(gan_path, transform=transform)
    real_dataset = ImageFolder(real_path, transform=transform)

    #Create dataloaders
    #Split into train/test
    gan_train_size = int(len(gan_dataset) * train_split)
    gan_test_size = len(gan_dataset) - gan_train_size
    gan_train_dataset, gan_test_dataset = random_split(
        gan_dataset, [gan_train_size, gan_test_size],
        generator=torch.Generator().manual_seed(42)  
    )

    real_train_size = int(train_split * len(real_dataset))
    real_test_size = len(real_dataset) - real_train_size
    real_train_dataset, real_test_dataset = random_split(
        real_dataset, [real_train_size, real_test_size],
        generator=torch.Generator().manual_seed(42)
    )

    class_weights = compute_class_weights(gan_dataset)


    # Create dataloaders
    if use_balanced_sampling:
        print('\n Using Balanced Sampling(WeightedRandomSampler)')
        gan_train_sampler = create_balanced_sampler(gan_train_dataset)

        gan_train_loader = DataLoader(
            gan_train_dataset, 
            batch_size=batch_size,
            sampler=gan_train_sampler,  # Use sampler instead of shuffle
            num_workers=num_workers
        )

        real_train_sampler = create_balanced_sampler(real_train_dataset)

        real_train_loader = DataLoader(
            real_train_dataset, 
            batch_size=batch_size,
            sampler=real_train_sampler, 
            num_workers=num_workers
        )

    else:
        print('\nUSING STANDARD RANDOM SAMPLING')
        # Training loaders: shuffle=True
        gan_train_loader = DataLoader(
            gan_train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=num_workers
        )
        real_train_loader = DataLoader(
            real_train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )

    # Test loaders: shuffle=False (for consistent evaluation), no sampling
    gan_test_loader = DataLoader(
        gan_test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    real_test_loader = DataLoader(
        real_test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    num_classes = len(gan_dataset.classes)
    class_names = gan_dataset.classes

    print(f'GAN dataset: {len(gan_dataset)} images')
    print(f'  - Train: {len(gan_train_dataset)} images')
    print(f'  - Test: {len(gan_test_dataset)} images')
    print(f'Real dataset: {len(real_dataset)} images')
    print(f'  - Train (unlabeled for DANN): {len(real_train_dataset)} images')
    print(f'  - Test (labeled for eval): {len(real_test_dataset)} images')
    print(f'Classes: {class_names}')

    return (gan_train_loader, gan_test_loader, 
            real_train_loader, real_test_loader,
            num_classes, class_names, class_weights)





# MODEL ARCHITECTURE

class FeatureExtractor(nn.Module):
    '''
    Remove final classification layer of ResNet18 to extract features
    '''

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained = True)
        # Remove final fc layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1) # Flatten: [batch, 512]
        return x

class Classifier(nn.Module):
    '''
    Classifier head on top of feature extractor
    '''
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, features):
        return self.fc(features)


class SourceModel(nn.Module):
    '''
    Complete Model: Feature Extractor + Classifier
    Trained only on GAN images
    '''
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)

        return output, features






# TRAINING SOURCE MODEL ON GAN IMAGES

def train_model(model, gan_loader, epochs, class_weights, lr = 0.001):
    '''
    Train classifier only on GAN images
    '''
    print('Training Source Model on GAN images only')

    model = model.to(device)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        #Track per-class accuracy
        class_correct = [0] * len(class_weights)
        class_total = [0] * len(class_weights)


        for images, labels in gan_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(len(class_weights)):
                class_mask = (labels == i)
                class_correct[i] += ((preds == labels) & class_mask).sum().item()
                class_total[i] += class_mask.sum().item()



    #Print per-class accuracy
    print('\nPer-class accuracy')
    for i in range(len(class_weights)):
        acc = class_correct[i] / class_total[i]
        print(f'Class Accuracy for {i} : {acc:.4f}')

    print(f'Source Model trained. Final accuracy on GAN : {correct/total:.4f}')
    return model




# EVALUATION

def evaluate_features_extract(model, dataloader, domain_name):
    '''
    Evaluate model on a given dataset
    Extract features from model's internal layers
    '''
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    features = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, ftrs = model(images)
            _, preds = outputs.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            features.append(ftrs.cpu().numpy())

        acc = correct/total
        features = np.vstack(features)

        print(f'{domain_name} acc : {acc:.4f}')

        return acc, np.array(all_preds), np.array(all_labels), features

def visualize(gan_features, real_features, gan_labels, real_labels, class_names,title):
    '''
    Visualize feature distributions using t-SNE
    Shows the domain gap between GAN and real features
    '''
    print('Generating t-SNE visualization')

    print(f'GAN features shape: {gan_features.shape}')
    print(f'Real features shape: {real_features.shape}')
    print(f'GAN labels: {np.bincount(gan_labels)}')
    print(f'Real labels: {np.bincount(real_labels)}')

    #Combine features
    all_features = np.vstack([gan_features, real_features])

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 30)
    features_2d = tsne.fit_transform(all_features)

    #Split back
    gan_2d = features_2d[:len(gan_features)]
    real_2d = features_2d[len(gan_features):]

    #Plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))

    #Plot 1: by domain (Real vs GAN)
    ax1.scatter(gan_2d[:,0], gan_2d[:,1], c = 'blue', alpha = 0.5, s = 20, label = 'GAN')
    ax1.scatter(real_2d[:,0], real_2d[:,1], c = 'red', alpha = 0.5, s = 20, label = 'REAL')
    ax1.set_title(f'{title} Feature Space - Domain Separation')
    ax1.legend()
    ax1.grid(alpha = 0.3)

    #Plot 2: by class
    colors = ['g','gold','m']

    #Check if valid labels

    if len(np.unique(real_labels)) <= 1:
        ax2.text(0.5, 0.5, 'No real labels available\n(using unlabeled data)', 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title(f'{title} Feature Space - Class Separation (No Labels)')
    else:
        for i, class_name in enumerate(class_names):
            #GAN samples of this class
            gan_mask = gan_labels == i
            if gan_mask.any():  # Check if mask has any True values
                ax2.scatter(gan_2d[gan_mask,0], gan_2d[gan_mask,1], c=colors[i], 
                            marker = 'o', alpha = 0.6, s = 30, label = f'{class_name} (GAN)',
                            edgecolors='black', linewidths=0.5)  

            #REAL samples of this class
            real_mask = real_labels == i
            if real_mask.any():  
                ax2.scatter(real_2d[real_mask,0], real_2d[real_mask,1], c=colors[i], 
                            marker = 'X', alpha = 0.8, s = 50, label = f'{class_name} (REAL)',
                            edgecolors='black', linewidths=0.5)  


        ax2.set_title(f'{title} Feature Space - Class Separation')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(alpha = 0.3)

    plt.tight_layout()
    plt.savefig(f'{title} feature_space_ppt.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    print('Saved feature_space.png')







# DOMAIN ADAPTATION

class GRL(torch.autograd.Function):
    '''
    Gradient Reversal Layer
    Forward: Identity
    Backward: Negate gradients (gradient = -gradient * lambda)
    This creates adversial training for domain confusion
    '''
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainClassifier(nn.Module):
    '''
    Domain Discriminator: Tries to distinguish between GAN and REAL
    '''

    def __init__(self, input_dim = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class DANNModel(nn.Module):
    '''
    Complete DANN architecture
    -Feature Extractor: Learns domain-invariant features
    -Class Classifier: Predicts class
    -Domain Classifier: Predicta domain (GAN = 0, Real = 1)
    '''
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.class_classifier = Classifier(num_classes)
        self.domain_classifier = DomainClassifier()

    def forward(self, x, alpha = 1.0):
        #Extract features
        features = self.feature_extractor(x)

        #Predict class
        class_output = self.class_classifier(features)

        #Gradient reversal and domain prediction
        reversed_features = GRL.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)

        return class_output, domain_output, features

def train_DANN(model, gan_loader, real_loader, epochs, class_weights, lr = 0.0001):
    '''
    Train model with DANN
    -Feature Extractor learns to fool the domain classifier
    -Domain Classifier tries to distinguish between REAL vs GAN
    -Result: Domain-invariant features
    '''
    print('Training with DANN Adaptation')

    model = model.to(device)


    class_weights = class_weights.to(device)
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    history = {
        'cls_loss': [],
        'dom_acc': [],
        'epochs': []

    }

    for epoch in range(epochs):
        model.train()
        running_cls_loss = 0.0
        running_dom_loss = 0.0

        domain_correct = 0
        domain_total = 0

        # Track per-class accuracy
        class_correct = [0] * len(class_weights)
        class_total = [0] * len(class_weights)

        #Alpha increases from 0 to 1 over training
        p = float(epoch)/epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        real_iter = iter(real_loader)
        num_batches = 0

        for gan_images, gan_labels in gan_loader:
            #Get Real images
            try:
                real_images, _ = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_images,_ = next(real_iter)

            batch_size = gan_images.size(0)
            gan_images = gan_images.to(device)
            real_images = real_images.to(device)
            gan_labels = gan_labels.to(device)

            #Domain labels: Real = 1, GAN = 0
            domain_labels_gan = torch.zeros(batch_size, 1).to(device)
            domain_labels_real = torch.ones(real_images.size(0), 1).to(device)

            #Forward pass on GAN and Real images
            class_output_gan, domain_output_gan, _ = model(gan_images, alpha)
            _, domain_output_real, _ = model(real_images, alpha)

            #Classification loss on GAN images
            cls_loss = criterion_class(class_output_gan, gan_labels)

            #Domain loss (both GAN and Real)
            dom_loss_gan = criterion_domain(domain_output_gan, domain_labels_gan)
            dom_loss_real = criterion_domain(domain_output_real, domain_labels_real)
            dom_loss = dom_loss_gan + dom_loss_real

            #Total Loss
            total_loss = cls_loss + dom_loss

            #Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            #Stats
            running_cls_loss += cls_loss.item()
            running_dom_loss += dom_loss.item()
            num_batches += 1

            with torch.no_grad():
                domain_preds_gan = (torch.sigmoid(domain_output_gan) > 0.5).float()
                domain_preds_real = (torch.sigmoid(domain_output_real) > 0.5).float()
                domain_correct += (domain_preds_gan == domain_labels_gan).sum().item()
                domain_correct += (domain_preds_real == domain_labels_real).sum().item()
                domain_total += domain_labels_gan.size(0) + domain_labels_real.size(0)



        #Calculate epochs average
        epoch_dom_loss = running_dom_loss / num_batches
        epoch_cls_loss = running_cls_loss / num_batches
        epoch_dom_acc = domain_correct / domain_total


        #Storing metrics
        history['cls_loss'].append(epoch_cls_loss)
        history['dom_acc'].append(epoch_dom_acc)
        history['epochs'].append(epoch + 1)

        # Warning if domain discriminator is not being fooled
        if epoch > epochs // 2 and epoch_dom_acc > 0.70:
            print('WARNING: Domain discriminator accuracy still high! GRL may not be working properly.')



    # Per-class accuracy
    print('\nPer-class accuracy:')
    for i in range(len(class_weights)):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f'Class Accuracy {i}: {class_acc:.4f}')



    print('Domain Adaptation complete')

    #Plot loss and accuracy
    plot_training_metrics(history)

    return model





def plot_training_metrics(history):
    '''
    Plot classification loss, domain loss, and domain accuracy
    '''

    epochs = history['epochs']
    cls_loss = history['cls_loss']
    dom_acc = history['dom_acc']

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize = (15,4))

    #Plot 1:  Classification Loss
    axes[0].plot(epochs, cls_loss, 'b-', linewidth=2, label='Classification Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha = 0.3)
    axes[0].legend()


    # Plot 3: Domain Accuracy
    axes[1].plot(epochs, dom_acc, 'g-', linewidth=2, label='Domain Accuracy')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Domain Discriminator Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('Training metrics.png', dpi = 300, bbox_inches='tight')
    print('\nTraining metrics plot saved')
    plt.close()










# DANN EVALUATION

class DANNEvaluator(nn.Module):
    def __init__(self, dann_model):
        super().__init__()
        self.dann_model = dann_model

    def forward(self, x):
        class_out,_,features = self.dann_model(x, alpha = 1.0)
        return class_out, features





from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def analyze_classification(true_labels, pred_labels, class_names):
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Real Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Detailed report
    print(classification_report(true_labels, pred_labels, 
                                target_names=class_names))

    # Per-class accuracy
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        class_acc = (pred_labels[class_mask] == i).mean()
        print(f'{class_name} accuracy: {class_acc:.2%}')





def main():
    '''
    Complete pipeline for domain adaptation experiment
    '''
    GAN_PATH = 'gan_images'
    REAL_PATH = 'real_images'
    BATCH_SIZE = 32
    EPOCHS_SOURCE = 10
    EPOCHS_ADAPT = 15


    print('Synthetic to Real Domain Adaptation')

    print('Loading datasets')

    (gan_train_loader, gan_test_loader, 
     real_train_loader, real_test_loader,
     num_classes, class_names, class_weights) = load_datasets(
        GAN_PATH, REAL_PATH, batch_size=BATCH_SIZE, train_split=0.8,use_balanced_sampling=True
    )
    print('Datasets loaded')

    print('\nTraining Source Model on GAN images only')
    model = SourceModel(num_classes)
    source_model = train_model(model, gan_train_loader, epochs=EPOCHS_SOURCE, class_weights=class_weights)
    print('\nSource Model trained on GAN images successfully')

    print('\nEvaluation of model')
    _, _, real_labels, real_features  = evaluate_features_extract(source_model, real_test_loader,domain_name = 'REAL(No Adaptation)')
    _, _, gan_labels, gan_features  = evaluate_features_extract(source_model, gan_test_loader,domain_name = 'GAN(No Adaptation)')
    print('\nEvaluation done')

    print('\nVisualization of Feature Space Before Adaptation')
    visualize(gan_features, real_features, gan_labels, real_labels,class_names,title = 'Before Adaptation')

    print('\nDANN Adaptation')
    dann_model = DANNModel(num_classes)
    dann_model = train_DANN(dann_model, gan_train_loader, real_train_loader,epochs = EPOCHS_ADAPT, class_weights=class_weights)

    print('\nFeatures Extraction after DANN')
    dann_eval = DANNEvaluator(dann_model)
    gan_acc_after, gan_preds, gan_labels_dann, gan_ftrs_dann = evaluate_features_extract(
        dann_eval, gan_test_loader, 'GAN Test (DANN)'
    )
    real_acc_after, real_preds, real_labels_dann, real_ftrs_dann = evaluate_features_extract(
        dann_eval, real_test_loader, 'REAL Test (DANN)'
    )
    visualize(gan_ftrs_dann, real_ftrs_dann, gan_labels_dann, real_labels_dann, 
             class_names, title='After Adaptation')

    print('\nDone.')



    # Check confusion matrix to see true performance
    print('\nConfusion Matrix')
    analyze_classification(real_labels_dann, real_preds, 
                          class_names)





if __name__=='__main__':
    main()








# In[ ]:




