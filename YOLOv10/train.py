from ultralytics import YOLO
import os
from pathlib import Path
import pickle

torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parse arguments
parser = argparse.ArgumentParser(description='YOLO Inference Script')
parser.add_argument('model_file', type=str, help='model file name including directory name')
args = parser.parse_args()
model = YOLO(args.model_file)


model.train(data='train_full.yml', epochs=300, batch=800, imgsz=224,device=DEVICE,val=False,verbose=True,cache=True,resume=False,half=False,
            freeze=[],save_period=2,exist_ok=True)
