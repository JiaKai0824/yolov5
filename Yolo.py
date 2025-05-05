import os
import shutil

os.environ["WANDB_DISABLED"] = "true"

IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 50

DATA_CONFIG = '"C:/Users/lojia/OneDrive/Documents/School/Degree/Year 3 Sem 1/TCV Computer Vision/Mango Dataset/yolov5/data.yaml"'
PRETRAINED_WEIGHTS = "yolov5s.pt"
RUN_NAME = "yolo_ripeness_model"

YOLO_DIR = r"C:\Users\lojia\OneDrive\Documents\School\Degree\Year 3 Sem 1\TCV Computer VIsion\Mango Dataset\yolov5"
os.chdir(YOLO_DIR)

train_command = f"python train.py --img {IMG_SIZE} --batch {BATCH_SIZE} --epochs {EPOCHS} --data {DATA_CONFIG} --weights {PRETRAINED_WEIGHTS} --name {RUN_NAME}"
os.system(train_command)

eval_command = f"python val.py --weights runs/train/{RUN_NAME}/weights/best.pt --data {DATA_CONFIG}"
os.system(eval_command)

best_model_path = f"runs/train/{RUN_NAME}/weights/best.pt"
saved_model_path = r"C:\yolov5_unprocess_model.pt"

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, saved_model_path)
    print(f"✅ Model saved as: {saved_model_path}")
else:
    print("❌ Training may have failed. No 'best.pt' file found!")

##python detect.py --weights runs/train/yolo_ripeness_model4/weights/best.pt --source "C:\Users\lojia\OneDrive\Documents\School\Degree\Year 3 Sem 1\TCV Computer VIsion\Mango Dataset\Merged_Dataset\Test\images" --img 512 --conf 0.5
