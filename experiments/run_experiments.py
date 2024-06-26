import time
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, random_split

from PerformerImageEncoder import PerformerImageEncoder
from experiments.experiment_utils import MLP, save_results
from utils import load_data


def performer_classification(
        task,
        train_data,
        test_data,
        performer_encoder_params,
        num_classes,
        batch_size,
        num_epochs,
        learning_rate,
        experiment_name,
        save_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """Perform classification using Performer encoder

    :param task: Task name
    :param train_data: Training data
    :param test_data: Testing data
    :param performer_encoder_params: Parameters for the Performer encoder
    :param num_classes: Number of classes
    :param batch_size: Batch size
    :param num_epochs: Number of epochs
    :param learning_rate: Learning rate
    :param experiment_name: Experiment name
    :param save_path: Path to save the results
    :param device: Device to run the experiment on
    """
    num_train = len(train_data)
    split = int(num_train * 0.9)  # 90% of data for training
    lengths = [split, num_train - split]
    train_subset, val_subset = random_split(train_data, lengths)

    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Performer encoder
    performer_encoder_params = deepcopy(performer_encoder_params)
    mlp_params = performer_encoder_params.pop('mlp_params')
    performer_encoder = PerformerImageEncoder(**performer_encoder_params)

    # Classifier
    classifier = MLP(
        input_dim=performer_encoder.linear_embedding_dim * performer_encoder.num_patches,
        output_dim=num_classes,
        **mlp_params
    ).to(device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(performer_encoder.performer.parameters()) + list(classifier.parameters()),
        lr=learning_rate
    )

    # Initialize learning rate scheduler
    if task == 'mnist':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif task == 'cifar':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    else:
        raise Exception('Task should be "mnist" or "cifar"!')

    # Save data
    training_info = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'test_losses': [],
        'test_accuracies': []
    }
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        performer_encoder.performer.train()
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            # to device
            data, targets = data.to(device), targets.to(device)
            # Zero the parameter gradients
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
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}, Time: {time.time() - start_time}')

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = round(100 * correct / total, 2)
        training_info['train_losses'].append(epoch_loss)
        training_info['train_accuracies'].append(epoch_accuracy)

        # Evaluation on validation set
        performer_encoder.performer.eval()
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                # to device
                data, targets = data.to(device), targets.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Encode batch
                encoded_data = performer_encoder.encode(data)
                # Flatten encoded data
                encoded_data_flat = encoded_data.view(encoded_data.size(0), -1)
                # Classify encoded data
                outputs = classifier(encoded_data_flat)
                # Calculate loss
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        # Calculate validation loss and accuracy
        val_loss = val_loss / len(val_loader)
        val_accuracy = round(100 * val_correct / val_total, 2)
        training_info['val_losses'].append(val_loss)
        training_info['val_accuracies'].append(val_accuracy)

        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                # to device
                data, targets = data.to(device), targets.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Encode batch
                encoded_data = performer_encoder.encode(data)
                # Flatten encoded data
                encoded_data_flat = encoded_data.view(encoded_data.size(0), -1)
                # Classify encoded data
                outputs = classifier(encoded_data_flat)
                # Calculate loss
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        # Calculate test loss and accuracy
        test_loss = test_loss / len(test_loader)
        test_accuracy = round(100 * test_correct / test_total, 2)
        training_info['test_losses'].append(test_loss)
        training_info['test_accuracies'].append(test_accuracy)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, '
            f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%'
        )

        scheduler.step()

    training_info['time'] = time.time() - start_time
    save_results(training_info, experiment_name, save_path)


def run_performer_experiment(task, performer_encoder_params, experiment_name, save_path, num_epochs=10, batch_size=32, learning_rate=1e-4):
    # Load dataset
    if task == 'mnist':
        train_data, test_data = load_data('mnist')['mnist']
    elif task == 'cifar':
        train_data, test_data = load_data('cifar')['cifar']
    else:
        raise Exception('Task should be "mnist" or "cifar"!')
    # Performer classification
    performer_classification(
        task, train_data, test_data,
        performer_encoder_params=performer_encoder_params,
        num_classes=10,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        experiment_name=experiment_name,
        save_path=save_path
    )