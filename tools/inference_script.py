from ultralytics import YOLOv10
import argparse
import os
import torch
import csv

def export_tensor(pt_path):
    model = YOLO(pt_path)
    model.export(
        imgsz=224,
        format="engine",
        dynamic=False,
        batch=8,
        workspace=10,
        half=True,
        int8=False,
        data="train_full_pseudo.yaml",
        device=[0,],
        opset=12,
    )

# Setup
torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser(description='YOLO Inference Script')
parser.add_argument('--model-file', type=str, help='model file name including directory name')
parser.add_argument('--source-path', type=str, help='Path to the directory containing images for inference')
parser.add_argument('--output-file', type=str, help='output CSV file name including directory name')
parser.add_argument('--engine', action='store_true', help='export tensorrt engine')
args = parser.parse_args()

# Load the YOLO model
model_path = args.model_file
if engine:
    if not os.path.exists(model_path.replace('.pt', '.engine')):
        print('start to export engine model')
        export_tensor(model_path)

    model_path = model_path.replace('.pt', '.engine')

net = YOLOv10(model_path,task='detect')

# Path to the directory containing images for inference
source_path = args.source_path


# Run inference on the images
results = net.predict(source=source_path, device=[0],conf=0.1,max_det=20,augment=False,imgsz=224,classes=[1,2,3,4],batch=8,half=True)

# Prepare the CSV file
csv_file_path = args.output_csv_file
with open(csv_file_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # csv_writer.writerow(['ImageId', 'PredictionString'])

    for result in results:
        # Get the image filename
        image_path = result.path
        image_name = os.path.basename(image_path)

        # Initialize an empty prediction string
        prediction_string = ""

        # Extract labels and bounding boxes
        labels = result.boxes.cls.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        # Construct the prediction string
        for label, box in zip(labels, boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            prediction_string += f"{int(label)} {x_min} {y_min} {x_max} {y_max} "

        # Write the row to the CSV file
        csv_writer.writerow([image_name, prediction_string.strip()])

print(f"Predictions saved to {csv_file_path}")


