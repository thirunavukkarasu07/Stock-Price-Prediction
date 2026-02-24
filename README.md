# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

---

## Problem Statement and Dataset

Stock price prediction is a challenging task due to the non-linear and volatile nature of financial markets. Traditional methods often fail to capture complex temporal dependencies. Deep learning, specifically **Recurrent Neural Networks (RNNs)**, can effectively model time-series dependencies, making them suitable for stock price forecasting.

* **Problem Statement**:
  Build an RNN model to predict the future stock price based on past stock price data.

* **Dataset**:
  A stock market dataset containing **historical daily closing prices** (e.g., Google, Apple, Tesla, or NSE/BSE data).
  The dataset is usually divided into **training and testing sets** after applying normalization and sequence generation.

  

---

<img width="681" height="316" alt="image" src="https://github.com/user-attachments/assets/98270a5e-cc03-4b16-bbbf-303d4bd7d29e" />


## Design Steps

### Step 1:

Import required libraries such as `torch`, `torch.nn`, `torch.optim`, `numpy`, `pandas`, and `matplotlib`.

### Step 2:

Load the dataset (e.g., stock closing prices from CSV), preprocess it by **normalizing** values between 0 and 1, and create input sequences for training/testing.

### Step 3:

Define the **RNN model architecture** with an input layer, hidden layers, and an output layer to predict stock prices.

### Step 4:

Compile the model using **MSELoss** as the loss function and **Adam optimizer**.

### Step 5:

Train the model on the training data, recording training losses for each epoch.

### Step 6:

Test the trained model on unseen data and visualize results by plotting the **true stock prices vs. predicted stock prices**.



## Program
#### Name:Thirunavukkarasu meenakshisundaram
#### Register Number:212224220117

```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(1, 64, 2, batch_first = True)
    self.fc = nn.Linear(64, 1)

  def forward(self,x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out





model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model

epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")







```
## Output

### True Stock Price, Predicted Stock Price vs time

Include your plot here

<img width="683" height="520" alt="image" src="https://github.com/user-attachments/assets/ef9ae572-2c7a-484f-b434-9dcc8f12e30a" />


### Predictions 

Include the predictions on test data

<img width="937" height="648" alt="image" src="https://github.com/user-attachments/assets/74994183-e86d-4af6-b6d7-f2c9ff97fa47" />


## Result

The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.

