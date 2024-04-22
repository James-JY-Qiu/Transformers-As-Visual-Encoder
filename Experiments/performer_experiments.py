import torch
from torch.utils.data import DataLoader

from PerformerEncoder import PerformerEncoder
from Experiments.experiment_utils import FullyConnectedLayer
from Experiments.experiment_params import MNIST_PERFORMER
from utils import load_data


def performer_classification(
        train_data,
        test_data,
        performer_params,
        num_classes,
        batch_size,
        num_epochs,
        learning_rate
):
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Performer encoder
    performer_encoder = PerformerEncoder(**performer_params)

    # Classifier
    classifier = FullyConnectedLayer(performer_encoder.linear_embedding_dim * performer_encoder.num_patches, num_classes)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(performer_encoder.performer.parameters()) + list(classifier.parameters()),
        lr=learning_rate
    )

    # Training loop
    for epoch in range(num_epochs):
        performer_encoder.performer.train()
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            # Encode batch
            encoded_data = performer_encoder.encode(data)
            # Flatten encoded data
            encoded_data_flat = encoded_data.view(encoded_data.size(0), -1)
            # Classify encoded data
            outputs = classifier(encoded_data_flat)
            # Calculate loss
            loss = criterion(outputs, targets)
            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')

    # Evaluation on test set
    performer_encoder.performer.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            # Encode batch
            encoded_data = performer_encoder.encode(data)
            # Classify encoded data
            outputs = classifier(encoded_data)
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')


def mnist_experiment(batch_size=64, num_epochs=10, learning_rate=1e-3):
    # Load MNIST data
    train_data, test_data = load_data('mnist')['mnist']
    # Performer classification
    performer_classification(
        train_data, test_data,
        MNIST_PERFORMER,
        num_classes=10,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )