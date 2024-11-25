import torch
import torch.nn as nn
from network import Discriminator
class ModifiedDiscriminator(nn.Module):
    def __init__(self, original_discriminator):
        super(ModifiedDiscriminator, self).__init__()
        self.features = nn.Sequential(*list(original_discriminator.children())[:-1])

    def forward(self, x):
        return self.features(x)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def extract_features(discriminator_model, train_loader, num_samples=0.1):
    discriminator_model.eval()
    features = []
    labels = []
    total_samples = len(train_loader.dataset)
    # get 10% data
    num_samples = int(total_samples * num_samples)
    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader):
            if i * len(data) >= num_samples:
                break
            feature = discriminator_model(data)
            feature = feature.view(-1, 512 * 4 * 4)
            features.append(feature)
            labels.append(target)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_classifier(features_train, labels_train, input_dim, learning_rate=0.001, epochs=100):
    classifier = LinearClassifier(input_dim=input_dim)
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(features_train)
        loss = criterion(output, labels_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return classifier


def evaluate_classifier(modified_discriminator, classifier, test_loader):
    classifier.eval()  # Set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            features = modified_discriminator(data)
            features = features.view(-1, 512 * 4 * 4)
            output = classifier(features)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# load data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained discriminator model
discriminator = Discriminator()
state_dict = torch.load('model_state_dict.pth', map_location=torch.device('cpu'))
discriminator.load_state_dict(state_dict)
modified_discriminator = ModifiedDiscriminator(discriminator)
modified_discriminator.to(device)
# Extract features from 10% of the training set
features_train, labels_train = extract_features(modified_discriminator, train_loader)

# Train the linear classifier
classifier = train_classifier(features_train, labels_train, input_dim=features_train.size(1))

# Evaluate on the test set
test_accuracy = evaluate_classifier(modified_discriminator, classifier, test_loader)

print(f"Test Accuracy: {test_accuracy:.2f}%")
