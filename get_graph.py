
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file with headers
df = pd.read_csv('facades_epoch_loss.csv')

# Create a figure with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

# Plot the 'epoch' column
df.plot(x='epoch', y='g_loss', ax=axes[0])
axes[0].set_title('Plot of Generator over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('L1 Loss')

# Plot the 'g_loss', 'cgan_loss', and 'd_loss' columns
df.plot(x='epoch', y=['d_loss'], ax=axes[1],color="Orange")
axes[1].set_title('Plot of Discriminator Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss Values')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("cityscapes_epoch_loss.png")
plt.show()

