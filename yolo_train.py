# %%
from ultralytics import YOLO
import argparse
import pandas as pd
import os
import sys

root_dir = os.getcwd()
sys.path.append(root_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True, help='Train the model')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--device', type=int, default=0, help='Device to use for training')
parser.add_argument('--lr0', type=float, default=0.0005, help='Initial learning rate')
parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate')
parser.add_argument('--dataset', type=str, default='wgisd', help='Dataset to use')


args, _ = parser.parse_known_args()

# Load model
model = YOLO('yolo12s.pt')  

if args.train:
    # Training
    results = model.train(data=f'Dataset_YOLO/{args.dataset}.yaml', 
                        batch = args.batch_size, 
                        device = args.device, 
                        name=args.dataset, 
                        save = True, 
                        epochs=args.epochs, 
                        lr0 = args.lr0, 
                        lrf = args.lrf)
else:
    # Load the model with the best weights
    model = YOLO(f'runs/detect/train_{args.dataset}/weights/best.pt')

    # Validation
    results = model.val(data=f"Dataset_YOLO/{args.dataset}.yaml", 
                        name = args.dataset, 
                        batch=args.batch_size, 
                        device=args.device, 
                        split ='test')

    # Saving metrics
    table_metrics = []
    for key in results.results_dict.keys():
        print(f"{key}: {results.results_dict[key]}")
        table_metrics.append([key, results.results_dict[key]])
    df = pd.DataFrame(table_metrics, columns=["Metric", "Value"])
    df.to_csv("wgisd_metrics.csv", index=False)

    # Testing
    results = model.predict(source=f"Dataset_YOLO/{args.dataset}/test/images", save=True)


# %%
