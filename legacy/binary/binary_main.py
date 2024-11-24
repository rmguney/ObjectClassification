# This file and it's contents are kept only as a reference for testing and learning purposes.
# They do not represent any part of the final project or its functionality.
# For the actual project please run main.py from the root directory instead.

import subprocess

def main():
    print("Select an option:")
    print("1. Test Dataset (Oxford Pets)")
    print("2. Test MobileNet")
    print("3. Test DETR")
    print("4. Train MobileNet")
    print("5. Train DETR")
    print("6. Run Classification Model Comparison")

    choice = input("Enter the number of your choice: ")
    if choice == "1":
        from src.test_pet_dataset import test_dataset
        test_dataset()
    elif choice == "2":
        from src.test_mobilenet import test_mobilenet
        test_mobilenet()
    elif choice == "3":
        from src.test_detr import test_detr
        test_detr()
    elif choice == "4":
        from src.train_mobilenet import train_mobilenet
        epochs = int(input("Enter number of epochs for MobileNet SSD: "))
        train_mobilenet(epochs=epochs)
    elif choice == "5":
        from src.train_detr import train_detr
        epochs = int(input("Enter number of epochs for DETR: "))
        train_detr(epochs=epochs)
    elif choice == "6":
        from src.compare_models import main as compare_models
        epochs = int(input("Enter number of epochs for legacy classification model comparison: "))
        compare_models(epochs=epochs)
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
