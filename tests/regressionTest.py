# Process the csv files in winequality.
# Run this file from root directory, not its directory:
# python ./tests/regressionTest.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class MyReg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(11, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
    
    def forward(self, x: torch.Tensor):
        # Expect x has shape [B, 11]
        
        x1 = torch.nn.functional.relu(self.fc1(x))
        x2 = torch.nn.functional.relu(self.fc2(x1))
        y = self.fc3(x2)

        return y # Shape [B, 1]


class MyDataset(Dataset):
    def __init__(self, whole_df: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        # Expect whole_df has shape [N, 11], labels has shape [N]
        self.data = whole_df
        self.labels = labels
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def transform_dataset(variant: str, force: bool=False) -> list[str]:
    # force=True enables rewrite of existing files.
    assert variant in ["white", "red"], "Datasets can be chosen from \"red\" and \"white\". "
    filepath = f"./datasets/winequality/winequality-{variant}.csv"
    out_train_path = filepath.replace(".csv", "-train.csv") # prepare output path.
    out_test_path = filepath.replace(".csv", "-test.csv") # prepare output path.
    if not force and os.path.isfile(out_train_path) and os.path.isfile(out_test_path):
        print("Dataset already transformed. If you wish to redo the transform, pass 'force=True'. ")
    else:
        df = pd.read_csv(filepath, sep=";") # Read with semicolon.

        # Below shows unbalance labels, from 3 to 9. 
        # print(np.unique(df["quality"], return_counts=True))
        feature_df = df.iloc[:, :-1]
        feature_mean = feature_df.mean()
        feature_std = feature_df.std()
        df.iloc[:, :-1] = (feature_df - feature_mean) / feature_std

        # Train test split
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=4896, 
                                                shuffle=True, stratify=df["quality"])
        # Make sure the split was stratified w.r.t. the objective.
        print(np.unique(df_train["quality"], return_counts=True))
        print(np.unique(df_test["quality"], return_counts=True))
        # Save into csv files. 
        df_train.to_csv(out_train_path)
        df_test.to_csv(out_test_path)
    return out_train_path, out_test_path


def csv_to_tensor(path: str):
    df = pd.read_csv(path)
    df.pop(df.columns[0]) # pop index column
    labels = df.pop(df.columns[-1])
    features = df
    # transform to tensor (assume there are N data).
    t_features = torch.tensor(features.values, dtype=torch.float32) # [N, 11]
    t_labels = torch.tensor(labels.values, dtype=torch.float32) # [N]
    return t_features, t_labels


def train(train_path: str):
    t_features, t_labels = csv_to_tensor(train_path)
    myDataset = MyDataset(t_features, t_labels)
    myDL = DataLoader(myDataset,batch_size=64,shuffle=True)
    print(myDL)
    # Build model and config
    model = MyReg()
    epochs = 1000
    lr = 5e-5
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # train model
    for epoch in range(epochs):
        if epoch % 10 == 9:
            print(f"Training epoch {epoch+1}...")
        running_loss = 0.
        for i, data in enumerate(myDL):
            inputs, labels = data # [B, 11], [B]
            B = labels.shape[0] # get batch size
            optimizer.zero_grad()
            outputs: torch.Tensor = model(inputs) # [B, 1]
            # MSE
            loss = (1/B) * torch.sum((torch.squeeze(outputs, dim=1) - labels) ** 2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= (i+1)
        print(f"    epoch loss is {running_loss} .")
            

    # save model
    torch.save(model.state_dict(), "./models/python-model.pth")

def test(test_path: str):
    pass

if __name__ == "__main__":
    os.makedirs("./models/", exist_ok=True)

    parser = argparse.ArgumentParser("Let user train or test linear regression model.")
    parser.add_argument('--variant', type=str, default="white")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--force', action=argparse.BooleanOptionalAction, default=False,
                            help="Force redo transform on dataset.")
    args = parser.parse_args()
    assert args.variant in ["white", "red"], "Variant can only be white or red. "
        
    # Transform dataset. 
    train_path, test_path = transform_dataset(variant = args.variant, force=args.force)

    # Train model.
    if args.train: train(train_path=train_path)
    else: test(test_path=test_path)