import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
# Hyper-parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
input_size = 784
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.01
# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(
    root='../data', train=False, transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, num_classes).to(device)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
t1 = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        # 新数组的shape属性应该要与原来数组的一致，即新数组元素数量与原数组元素数量要相等。
        # 一个参数为-1时，那么reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值。
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, total_step, loss.item()))
t2 = time.time()
print(float(t2 - t1) / 60)
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # orch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))

# Save the model checkpoint
torch.save(model, 'Logisticmodel.ckpt')