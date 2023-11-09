import matplotlib.pyplot as plt

# FFNN Data
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_acc = [0.533625, 0.58375, 0.61775, 0.642, 0.654875, 0.677125, 0.703, 0.705375, 0.737, 0.757625]  # Replace with your actual training accuracy values
val_acc = [0.5225, 0.56625, 0.60375, 0.5625, 0.63125, 0.60875, 0.5575, 0.58625, 0.62125, 0.6025]  # Replace with your actual validation accuracy values

# RNN Data
# epochs = [1, 2, 3, 4, 5]
# train_acc = [0.4685, 0.53675, 0.566625, 0.588625, 0.595125]  # Replace with your actual training accuracy values
# val_acc = [0.4825, 0.51875, 0.58375, 0.61125, 0.58125]  # Replace with your actual validation accuracy values

# Training loss data
# train_loss = [0.7950025200843811, 0.9639660120010376, 0.8364649415016174, 1.003187656402588, 0.5189852714538574, 0.38564950227737427, 0.6680692434310913, 0.7551859617233276, 0.5097081065177917, 0.7299635410308838]
# train_loss = [1.0668243169784546, 1.014032244682312, 0.9026938080787659, 0.9121338725090027, 0.88104248046875]

# Create the plot
plt.figure(figsize=(10, 6))  
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', marker='o', linestyle='-', label='Validation Accuracy')
#Add this line for training loss plot, and remove above two lines of accuracy plot code
# plt.plot(epochs, train_loss)
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()




