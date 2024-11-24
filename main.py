import subprocess

def main():
    print("Select an option:")
    print("1. Test Dataset (CIFAR-10)")
    print("2. Train MobileNet")
    print("3. Train DETR")
    print("4. Run Comparison")

    choice = input("Enter the number of your choice: ")

    if choice == "1":
        from src.test_cifar10_dataset import test_cifar10_dataset
        test_cifar10_dataset()

    elif choice == "2":
        from src.models.train_mobilenet import train_mobilenet
        from src.cifar10_dataset import get_cifar10_dataloaders

        dataloaders = get_cifar10_dataloaders(batch_size=16)
        print("Dataloader Structure: ", type(dataloaders), dataloaders)

        epochs = int(input("Enter number of epochs for MobileNet training: "))
        learning_rate = float(input("Enter learning rate for MobileNet: "))

        # Handle different return types
        if isinstance(dataloaders, tuple):
            if len(dataloaders) == 2:
                train_loader, val_loader = dataloaders
            elif len(dataloaders) == 3:
                train_loader, val_loader, test_loader = dataloaders
            else:
                raise ValueError("Unexpected number of values returned by get_cifar10_dataloaders.")
        elif isinstance(dataloaders, dict):
            train_loader = dataloaders.get("train")
            val_loader = dataloaders.get("val")
            test_loader = dataloaders.get("test")
        else:
            raise ValueError("get_cifar10_dataloaders returned an unsupported type.")

        train_mobilenet(train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)

    elif choice == "3":
        from src.models.train_detr import train_detr
        from src.cifar10_dataset import get_cifar10_dataloaders

        dataloaders = get_cifar10_dataloaders(batch_size=16)
        print("Dataloader Structure: ", type(dataloaders), dataloaders)

        epochs = int(input("Enter number of epochs for DETR (Tuned) training: "))
        learning_rate = float(input("Enter learning rate for DETR (Tuned): "))

        # Handle different return types
        if isinstance(dataloaders, tuple):
            if len(dataloaders) == 2:
                train_loader, val_loader = dataloaders
            elif len(dataloaders) == 3:
                train_loader, val_loader, test_loader = dataloaders
            else:
                raise ValueError("Unexpected number of values returned by get_cifar10_dataloaders.")
        elif isinstance(dataloaders, dict):
            train_loader = dataloaders.get("train")
            val_loader = dataloaders.get("val")
            test_loader = dataloaders.get("test")
        else:
            raise ValueError("get_cifar10_dataloaders returned an unsupported type.")

        train_detr(train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)

    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
