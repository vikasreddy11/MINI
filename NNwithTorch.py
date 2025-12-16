import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_datasets=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_datasets=datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_datasets,batch_size=32,shuffle=True)
test_loader=DataLoader(test_datasets,batch_size=32,shuffle=False)

class NeuralN(nn.Module):
    def __init__(self):
        super(NeuralN,self).__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
model=NeuralN()

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.02)

epochs=10

for epoch in range(epochs):
    model.train()
    running_loss=0.0

    for imgs,lab in  train_loader:

        optimizer.zero_grad()

        outputs=model(imgs)

        loss=criterion(outputs,lab)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

model.eval()
correct=0
total=0

with torch.no_grad():  # Disable gradient calculation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (100 * correct) / total
print(f'Test Accuracy: {accuracy:.2f}%')

images, labels = next(iter(test_loader))


model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

print(f'Predicted: {predicted[:10]}')
print(f'Actual: {labels[:10]}')