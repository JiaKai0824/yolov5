import pandas as pd
import matplotlib.pyplot as plt

# Load training results from YOLO logs (CSV file)
results_path = "runs/train/yolo_ripeness_model5/results.csv"
df = pd.read_csv(results_path)

# Extract columns based on your CSV structure
epochs = df["               epoch"]
train_loss = df["      train/box_loss"]  # Training loss
val_mAP50 = df["     metrics/mAP_0.5"]  # Validation accuracy (mAP@50)

# Plot Accuracy (mAP50)
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_mAP50, label="Validation mAP50", marker="o", color="blue")
plt.xlabel("Epochs")
plt.ylabel("mAP@50")
plt.title("YOLO Model Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()

# Plot Training Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Training Loss", marker="o", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("YOLO Model Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()
