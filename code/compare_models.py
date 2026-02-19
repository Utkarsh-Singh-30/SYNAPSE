import argparse
from src.trainer import train_one_model

# The list of 6 models to compare for your thesis
MODELS_TO_COMPARE = [
    'densenet121',
    'densenet169',
    'resnet101',
    'efficientnet_b1',
    'seresnext50_32x4d',
    'thoraxnet'
]
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs per model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    print("==================================================")
    print("   STARTING COMPARISON OF 6 ARCHITECTURES")
    print("==================================================")
    print(f"Models: {MODELS_TO_COMPARE}")
    print(f"Epochs: {args.epochs}")
    print("==================================================\n")

    for model_name in MODELS_TO_COMPARE:
        try:
            train_one_model(model_name, epochs=args.epochs, batch_size=args.batch_size)
        except Exception as e:
            print(f"\n[ERROR] Failed to train {model_name}: {e}")
            print("Skipping to next model...\n")

    print("\n\n##################################################")
    print("   COMPARISON SUITE COMPLETE")
    print("   Check 'new_output/plots/' for logs and results.")
    print("##################################################")