from ultralytics import YOLO
import os
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='YOLO Inference Script')
parser.add_argument('model_file', type=str, help='model file name including directory name')
parser.add_argument('source_path', type=str, help='Path to the directory containing images for inference')
parser.add_argument('output_file', type=str, help='output file name including directory name')
args = parser.parse_args()

torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# Load the exported TensorRT model
model = YOLO(args.model_file, task="detect")
countries = ['Japan', 'China_MotorBike', 'Czech', 'India', 'Norway',
             'United_States', ]
task_path = Path(args.output_file).parent
task_num_path = str(task_path / f'countries')

os.makedirs(task_num_path, exist_ok=True)
with open(args.output_file, 'w') as fw:
    for country in countries:
        if country == 'Norway':
            results = model(args.source_path + f'/{country}/test/crop/images', conf=0.1, device=DEVICE, max_det=20,
                            augment=True, plots=True, batch=8,
                            imgsz=640, classes=[1, 2, 3, 4])
            with open(task_num_path + f'/{country}_results.txt', 'w') as f:
                for r in results:
                    txt = f'{Path(r.path).stem}.jpg,'
                    init_len = len(txt)
                    for idx, (cls, xyxy) in enumerate(
                            zip(r.boxes.cls, r.boxes.xyxy)):
                        xyxy = xyxy.cpu().numpy()
                        xyxy[3] += 1000
                        xyxy[1] += 1000
                        if idx == 0:
                            txt += f'{cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                        else:
                            txt += f' {cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                    f.write(txt + '\n')
                    fw.write(txt + '\n')
        else:
            results = model(args.source_path + f'/{country}/test/images', conf=0.1, device=DEVICE, max_det=20,
                            augment=True, plots=True, batch=8,
                            imgsz=640, classes=[1, 2, 3, 4])
            with open(task_num_path + f'/{country}_results.txt', 'w') as f:
                for r in results:
                    txt = f'{Path(r.path).stem}.jpg,'
                    init_len = len(txt)
                    for idx, (cls, xyxy) in enumerate(
                            zip(r.boxes.cls, r.boxes.xyxy)):
                        xyxy = xyxy.cpu().numpy()
                        if idx == 0:
                            txt += f'{cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                        else:
                            txt += f' {cls.cpu().numpy().astype(int)} {" ".join(map(str, xyxy.astype(int)))}'
                    f.write(txt + '\n')
                    fw.write(txt + '\n')
