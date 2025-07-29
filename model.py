import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Model will run on device: {device}")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """
        Initializes the Linear_QNet neural network.

        Args:
            input_size (int): Number of input features (state representation).
            hidden_size1 (int): Number of neurons in the first hidden layer.
            hidden_size2 (int): Number of neurons in the second hidden layer.
            output_size (int): Number of output actions (Q-values for each action).
        """
        super().__init__()
        # Define the neural network layers
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        
        # Move the model to the specified device (CPU/GPU)
        self.to(device)

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor representing the game state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = torch.relu(self.linear1(x)) # Apply ReLU activation after first layer
        x = torch.relu(self.linear2(x)) # Apply ReLU activation after second layer
        x = self.linear3(x)             # Output layer (no activation for Q-values)
        return x

    def save(self, file_name='model.pth'):
        """
        Saves the model weights to disk.
        Note: In our current setup, agent.py handles saving both main and target models.
              This method is kept for completeness but might not be directly called.
        """
        model_folder_path = 'model_weights' # Ensure this matches the directory in agent.py
        os.makedirs(model_folder_path, exist_ok=True)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"🧠 Model saved to {file_path}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        """
        Initializes the Q-learning trainer.

        Args:
            model (nn.Module): The Q-network (main network).
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model # The main Q-network
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Error loss
        # Learning rate scheduler: reduces LR by gamma (0.9) every step_size (50) epochs
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.9)
        self.model.to(device) # Ensure the model is on the correct device

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        """
        Performs a single training step for the Q-network.
        MODIFIED to return the loss value for plotting.
        """
        # Convert all numpy arrays to PyTorch tensors and move to the device
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)

        # Handle single sample vs. batch: Unsqueeze if input is a single sample
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,) # Convert single boolean to a tuple for iteration

        # 1. Predict Q values for the current state using the main model
        pred = self.model(state)

        # Clone predictions to create target for modification
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx] # Initialize Q_new with immediate reward

            # If the episode is not done, calculate Q_new using the Bellman equation
            if not done[idx]:
                # Use the target model for Q-value of the next state (for stability)
                if target_model:
                    with torch.no_grad(): # Ensure no gradients for target Q-value calculation
                        Q_next = target_model(next_state[idx].unsqueeze(0))
                    Q_new = reward[idx] + self.gamma * torch.max(Q_next)
                else:
                    # Fallback to using the main model if target_model is not provided (less stable)
                    with torch.no_grad():
                        Q_next = self.model(next_state[idx].unsqueeze(0))
                    Q_new = reward[idx] + self.gamma * torch.max(Q_next)

            # Update the target Q-value for the action that was actually taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Backpropagation:
        self.optimizer.zero_grad() # Clear gradients from previous step
        loss = self.criterion(target, pred) # Calculate MSE loss between target and predicted Q-values
        loss.backward() # Compute gradients
        self.optimizer.step() # Update model weights
        self.scheduler.step() # Update the learning rate based on the scheduler

        # MODIFICATION: Return the scalar value of the loss for metrics tracking
        return loss.item()
