import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import MLP
import torchmetrics
import logging

MLP_input_size=39
MLP_hidden_size=200
MLP_output_size=1
batch_size = 1024
learning_rate = 0.001
num_epochs = 100
torch.manual_seed(42)
logging.basicConfig(filename='./training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

x_train_path="./undersampling_X_train.csv"
x_test_path="./undersampling_X_test.csv"
y_train_path="./undersampling_Y_train.csv"
y_test_path="./undersampling_Y_test.csv"

x_train=torch.tensor(np.loadtxt(x_train_path,delimiter=",",dtype=float)).float()
y_train=torch.tensor(np.loadtxt(y_train_path,delimiter=",",dtype=int)).float()
x_test=torch.tensor(np.loadtxt(x_test_path,delimiter=",",dtype=float)).float()
y_test=torch.tensor(np.loadtxt(y_test_path,delimiter=",",dtype=int)).float()
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=MLP_input_size, hidden_size=MLP_hidden_size, output_size=MLP_output_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

accuracy = torchmetrics.Accuracy(task="BINARY").to(device)
precision = torchmetrics.Precision(task="BINARY").to(device)
recall = torchmetrics.Recall(task="BINARY").to(device)
f1 = torchmetrics.F1Score(task="BINARY").to(device)

def evaluate_model(loader, model):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total = 0
    correct_predictions = 0
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            preds = outputs > 0.5
            targets=targets.unsqueeze(1)
            correct_predictions += (preds == targets).sum().item()
            loss = criterion(outputs, targets)
            total_loss += loss.item() 
            total += data.size(0)
            accuracy.update(preds, targets.int())
            precision.update(preds, targets.int())
            recall.update(preds, targets.int())
            f1.update(preds, targets.int())
    model.train()  # Set the model back to training mode
    return total_loss / total, accuracy.compute(), precision.compute(), recall.compute(), f1.compute()

for epoch in range(num_epochs):
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets.unsqueeze(1))

        # Backward and optimize
        # print(f'Epoch [{epoch+1}/{num_epochs}], Batch train Loss: {loss:.4f}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save(model.state_dict(), "./checkpoint/model_checkpoint_{epoch}.pth".format(epoch=epoch+1))
    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate_model(test_loader, model)
    # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Prec: {test_prec:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Prec: {test_prec:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

print("Training complete.")