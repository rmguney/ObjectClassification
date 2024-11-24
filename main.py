def get_dataloaders(batch_size):
    """
    Helper function to create and handle Pascal VOC dataloaders.
    """
    from src.dataset.initialize_pascalvoc import create_dataloaders

    dataloaders = create_dataloaders(batch_size=batch_size)
    print("Dataloader Structure: ", type(dataloaders))

    # Handle different return types
    if isinstance(dataloaders, tuple):
        if len(dataloaders) == 2:
            return dataloaders[0], dataloaders[1]
        elif len(dataloaders) == 3:
            return dataloaders[0], dataloaders[1], dataloaders[2]
        else:
            raise ValueError("Unexpected number of values returned by create_dataloaders.")
    elif isinstance(dataloaders, dict):
        train_loader = dataloaders.get("train")
        val_loader = dataloaders.get("val")
        return train_loader, val_loader
    else:
        raise ValueError("create_dataloaders returned an unsupported type.")

def main():
    print("Select an option:")
    print("1. Initialize Pascal VOC Dataset")
    print("2. Test Pascal VOC Dataset")
    print("3. Train MobileNet")
    print("4. Train DETR")

    try:
        choice = int(input("Enter the number of your choice: "))
    except ValueError:
        print("Invalid input! Please enter a number between 1 and 4.")
        return

    if choice == 1:
        from src.dataset.initialize_pascalvoc import download_and_prepare_voc
        download_and_prepare_voc()
        print("Pascal VOC dataset initialized successfully.")

    elif choice == 2:
        from src.dataset.test_pascalvoc import test_pascalvoc_dataset
        test_pascalvoc_dataset()

    elif choice == 3:
        from src.models.train_mobilenet import train_mobilenet

        batch_size = int(input("Enter batch size for MobileNet training: "))
        train_loader, val_loader = get_dataloaders(batch_size)

        epochs = int(input("Enter number of epochs for MobileNet training: "))
        learning_rate = float(input("Enter learning rate for MobileNet: "))

        train_mobilenet(train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)

    elif choice == 4:
        from src.models.train_detr import train_detr

        batch_size = int(input("Enter batch size for DETR training: "))
        train_loader, val_loader = get_dataloaders(batch_size)

        epochs = int(input("Enter number of epochs for DETR training: "))
        learning_rate = float(input("Enter learning rate for DETR: "))  # Ensure learning rate is a float

        train_detr(train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)

    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
