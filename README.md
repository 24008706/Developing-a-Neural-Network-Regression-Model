# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1057" height="702" alt="image" src="https://github.com/user-attachments/assets/0560e8cd-f695-456b-956d-2e6612864f95" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:MUTHUREVULA SAHITHI

### Register Number:212224040208

```
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')



```

### Dataset Information
<img width="192" height="272" alt="image" src="https://github.com/user-attachments/assets/25733c16-1262-4a34-b5ea-71e2f5940475" />

### OUTPUT
<img width="683" height="411" alt="image" src="https://github.com/user-attachments/assets/dfcedb81-47c4-4c6b-ae52-65c1d2de0609" />

### Training Loss Vs Iteration Plot
<img width="840" height="575" alt="image" src="https://github.com/user-attachments/assets/147604c4-f1f7-43ec-b99b-811ea627c25c" />

### New Sample Data Prediction

<img width="1042" height="126" alt="image" src="https://github.com/user-attachments/assets/30d71e39-8aae-4694-91c8-2882d1075d1e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
