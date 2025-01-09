import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def train_nn(X_train, X_test, y_train, y_test, print_graph=False):
    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(NeuralNetwork, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )

        def forward(self, x):
            return self.model(x)

    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy().reshape(-1, 1)
    y_test_np = y_test.to_numpy().reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_np)
    X_val_scaled = scaler_X.transform(X_test_np)

    y_train_scaled = scaler_y.fit_transform(y_train_np)
    y_val_scaled = scaler_y.transform(y_test_np)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    input_dim = X_train.shape[1]
    output_dim = 1
    model = NeuralNetwork(input_dim, output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 5000
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions_train = model(X_train_tensor)
        train_loss = criterion(predictions_train, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions_val = model(X_val_tensor)
            val_loss = criterion(predictions_val, y_val_tensor)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

        # if (epoch + 1) % 50 == 0:
        #     print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    torch.save(best_model_state, "best_model.pth")
    print(f"\nBest model saved from epoch {best_epoch} with validation loss {best_val_loss:.4f}")

    if print_graph:
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Over Epochs')
        plt.show()

    model.load_state_dict(torch.load("best_model.pth",weights_only=True))
    model.eval()

    with torch.no_grad():
        y_pred_tensor = model(X_val_tensor)
        y_pred = scaler_y.inverse_transform(y_pred_tensor.numpy())
        y_val_original = scaler_y.inverse_transform(y_val_tensor.numpy())

    return y_pred, y_val_original