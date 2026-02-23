# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
To develop a deep learning-based classification model using transfer learning with the VGG-19 architecture to accurately classify images of capacitors into two categories: defected and non-defected. The model will be trained on labeled images, optimized using appropriate loss and optimizer functions, and evaluated using metrics such as confusion matrix, classification report, and performance plots.
<br>
This classification system helps automate the quality control process in manufacturing by detecting defective capacitors based on their images, thereby improving efficiency and reducing manual inspection errors. The project leverages pre-trained VGG-19 weights and adapts the final classification layer to suit the specific dataset.
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
Import the necessary libraries such as PyTorch, Torchvision, and Matplotlib.  


### STEP 2:
Load the dataset and apply preprocessing (resizing, normalization, and augmentation).  


### STEP 3:
Download the pre-trained VGG-19 model from Torchvision models.  


### STEP 4:
Freeze the feature extraction layers of VGG-19.  


### STEP 5:
Modify the final fully connected layer to match the number of dataset classes.  


### STEP 6:
Define the loss function (CrossEntropyLoss) and optimizer (Adam/SGD).  


### STEP 7:
Train the model on the training dataset and validate on the validation set.  


### STEP 8:
Plot Training Loss and Validation Loss vs Iterations.  


### STEP 9:
Evaluate the model on the test dataset.  


### STEP 10:
Generate Confusion Matrix, Classification Report, and test on new sample images.  

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(pretrained=True)

# Modify the final fully connected layer to match the dataset classes

in_features=model.classifier[-1].in_features
num_classes = len(train_dataset.classes)
model.classifier[-1] = nn.Linear(in_features, 1)

# Include the Loss function and optimizer

model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)


# Train the model

def train_model(model, train_loader, test_loader, num_epochs=5):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # Ensure proper shape for BCEWithLogitsLoss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: JANANI S")
    print("Register Number: 212223230086")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="850" height="755" alt="image" src="https://github.com/user-attachments/assets/c48c3eda-d94d-406f-aa16-4cd0fcc13849" />

</br>
</br>
</br>

### Confusion Matrix
<img width="829" height="676" alt="image" src="https://github.com/user-attachments/assets/68f86b21-f7af-480a-942d-cd7e5893657a" />

</br>
</br>
</br>

### Classification Report
<img width="1271" height="428" alt="image" src="https://github.com/user-attachments/assets/0fa97468-1e13-4875-b86a-7fa71575d598" />

</br>
</br>
</br>

### New Sample Prediction
<img width="511" height="439" alt="image" src="https://github.com/user-attachments/assets/4da0d685-f6ba-49fc-abc5-44b0e5cbaf13" />

<img width="483" height="448" alt="image" src="https://github.com/user-attachments/assets/5f8a7d34-93dc-487d-b1c0-10837b9a14eb" />

</br>
</br>
</br>

## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
</br>
</br>
</br>
