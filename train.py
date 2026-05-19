import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet import ResNet18, ResNet50, ResNeXt50_32x4d


def get_model(model_name, num_classes, pretrained=True):
    """Return a ResNet model with the final FC layer replaced for num_classes."""
    models = {
        "resnet18": ResNet18,
        "resnet50": ResNet50,
        "resnext50": ResNeXt50_32x4d,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    model = models[model_name](pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_data_loaders(data_dir, batch_size, num_workers=4):
    """Return CIFAR-10 train and test DataLoaders with standard transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch and print progress."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "resnext50"],
                        help="Model architecture (default: resnet18)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (default: 5e-4)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory for CIFAR-10 data (default: ./data)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints (default: ./checkpoints)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size,
                                                  args.num_workers)
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset)}")

    # Model
    num_classes = 10  # CIFAR-10
    model = get_model(args.model, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print(f"Model: {args.model} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    best_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Test Loss':>9} | {'Test Acc':>8} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2f}% | "
              f"{test_loss:>9.4f} | {test_acc:>7.2f}% | {lr:>8.6f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
            }, save_path)

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Best checkpoint saved to: {os.path.join(args.save_dir, args.model + '_best.pth')}")


if __name__ == "__main__":
    main()
