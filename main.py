# Importing the required Libraries
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.io import read_image
from sklearn.metrics import confusion_matrix, f1_score
from PIL import Image

SAVED_MODELS_FOLDER_PATH = 'saved_models'
PRETRAINED_WEIGHTS_FILE_NAME = 'resnet34-b627a593'
FINE_TUNED_MODEL = 'trained_model'

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.images = []
        for i, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.images.append((img_path, i))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_idx = self.images[idx]
        img = read_image(img_path).float()
        if self.transform:
            img = self.transform(img)
        return img, class_idx
    


def get_prediction(_net, _test_image):
    net.eval()
    with torch.no_grad():
        output = net(_test_image)
        probabilities = nn.functional.softmax(output, dim=1)[0]
        _predicted_label = torch.argmax(probabilities).item()
        return _predicted_label


def print_evaluation_results(loss, accuracy, confusion_mat, f1):
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print(f"F1 Score: {f1:.4f}")


def plot_confusion_matrix(confusion_mat, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet34():
    layers = [3, 4, 6, 3]
    model = ResNet(BasicBlock, layers)
    return model


def evaluate(model, data_loader, criterion, device):
    running_loss = 0.0
    correct_predictions = 0
    _total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            _total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    loss = running_loss / _total_samples
    accuracy = correct_predictions / _total_samples
    confusion_mat = confusion_matrix(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return loss, accuracy, confusion_mat, f1


def train(model, _train_loader, _val_loader, _num_epochs, _learning_rate, _criterion, _optimizer):
    model.to(DEVICE)

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    for epoch in range(_num_epochs):
        model.train()  # Set the model to train mode
        running_loss = 0.0
        correct_predictions = 0
        _total_samples = 0

        for images, labels in _train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            _optimizer.zero_grad()
            outputs = model(images)
            loss = _criterion(outputs, labels)
            loss.backward()
            _optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            _total_samples += labels.size(0)

        train_loss = running_loss / _total_samples
        train_accuracy = correct_predictions / _total_samples

        # Validate the model
        model.eval()  # Set the model to evaluation mode
        val_loss, val_accuracy, cf, f1 = evaluate(model, _val_loader, _criterion, DEVICE)

        print(f'Epoch [{epoch + 1}/{_num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    return train_losses, train_accuracies, val_losses, val_accuracies


def save_net(_net, path, file_name):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(_net.state_dict(), str(path + '/' + file_name + '.pth'))


def load_net(path, file_name):
    file_path = str(path + '/' + file_name + '.pth')
    if not os.path.isfile(file_path):
        raise Exception('No models found at path: ' + file_path)
    else:
        return torch.load(file_path)

class ConvertToRGB(object):
    def __call__(self, img):
        # Check if the image is grayscale (single channel)
        if img.size(0) == 1:
            # Convert grayscale to RGB by repeating the single channel
            img = torch.cat([img, img, img], dim=0)
        return img


if __name__ == '__main__':

    if torch.cuda.is_available():
        DEVICE_NAME = torch.cuda.get_device_name(0)
        torch.cuda.empty_cache()
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    print(f'USING DEVICE: {DEVICE_NAME}')

    if len(sys.argv) > 1:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            ConvertToRGB(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = CustomDataset(sys.argv[1], transform=transform)

        total_samples = dataset.__len__()
        classes = dataset.classes
        total_classes = len(classes)
        print('TOTAL SAMPLES:\t', total_samples)
        print('CLASSES:\t', classes)
        print('TOTAL CLASSES:\t', total_classes)

        criterion = nn.CrossEntropyLoss()
        net = resnet34()

        pretrained_weights = load_net(SAVED_MODELS_FOLDER_PATH, PRETRAINED_WEIGHTS_FILE_NAME)

        net.load_state_dict(pretrained_weights)

        for param in net.parameters():
            param.requires_grad = False

        for param in net.layer4.parameters():
            param.requires_grad = True

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 7)


        if len(sys.argv) > 2 and sys.argv[2] == 'train':
            train_size = 0.80
            val_size = 0.20
            train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

            # Hyper Parameters
            batch_size = 64
            learning_rate = 0.01
            momentum = 0.9
            num_epochs = 10
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

            train_losses, train_accuracies, val_losses, val_accuracies = train(net, train_loader, val_loader,
                                                                               num_epochs, learning_rate, criterion,
                                                                               optimizer)

            epochs = np.linspace(1, num_epochs, num_epochs)
            plt.title('Loss')
            plt.plot(epochs, train_losses, label='Train')
            plt.plot(epochs, val_losses, label='Train')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            epochs = np.linspace(1, num_epochs, num_epochs)
            plt.title('Accuracy')
            plt.plot(epochs, train_accuracies, label='Train')
            plt.plot(epochs, val_accuracies, label='Train')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

            loss, accuracy, confusion_mat, f1 = evaluate(net, train_loader, criterion, DEVICE)

            print_evaluation_results(loss, accuracy, confusion_mat, f1)
            plot_confusion_matrix(confusion_mat, classes=classes)

            save_net(net, SAVED_MODELS_FOLDER_PATH, FINE_TUNED_MODEL)

        elif len(sys.argv) > 2 and sys.argv[2] == 'test':

            learned_weights = load_net(SAVED_MODELS_FOLDER_PATH, FINE_TUNED_MODEL)
            net.load_state_dict(learned_weights)
            net.to(DEVICE)
            batch_size = 32
            dataset_path = sys.argv[1]
            test_set = CustomDataset(dataset_path, transform)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
            loss, accuracy, confusion_mat, f1 = evaluate(net, test_loader, criterion, DEVICE)

            print_evaluation_results(loss, accuracy, confusion_mat, f1)
            plot_confusion_matrix(confusion_mat, classes=classes)

        elif len(sys.argv) > 2 and sys.argv[2] == 'inference':
            files = os.listdir()
            test_image_path = ''
            for i in files:
                if 'prediction_image' == i[:-5]:
                    test_image_path = i
                    break

            if not os.path.isfile(test_image_path):
                print('Place an image in the current directory with name : \"prediction_image.png\"')
                sys.exit(0)

            # Preprocess the test image
            test_image = Image.open(test_image_path)
            test_image_preprocessed = transform(test_image)
            input_image = test_image_preprocessed.to(device=DEVICE).unsqueeze(dim=0)
            net_state = load_net(SAVED_MODELS_FOLDER_PATH, FINE_TUNED_MODEL)
            net.load_state_dict(net_state)
            net.to(device=DEVICE)
            predicted_label = get_prediction(net, input_image)

            fig, axes = plt.subplots(1, 2)
            axes[0].set_title('Test Image')
            axes[0].imshow(test_image, interpolation='nearest')
            axes[1].set_title(str('Predicted Label: ' + str(dataset.classes[predicted_label])))
            axes[1].imshow(test_image_preprocessed.permute(1, 2, 0), interpolation='nearest')
            plt.tight_layout()
            plt.show()
