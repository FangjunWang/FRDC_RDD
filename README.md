

# Road Damage Detection Competition

This repository contains the code and models for road damage detection, submitted for the competition. The code is structured into four main directories, each containing various models and tools used in the detection and processing pipeline.

## Directory Structure

- **Co-DERT**: Contains model configuration and training scripts based on MMDetection 3.3.0.
- **RTMDet**: Includes configuration files and training scripts for RTMDet, also based on MMDetection 3.3.0.
- **YOLOv10**: Implements the YOLOv10 model, based on the latest version of Ultralytics.
- **tools**: Utility scripts for processing images, converting annotations, and exporting models to TensorRT.
- **annotations**: Contains dataset split with COCO type annotations, pseudo labels.

## Dependencies

- **MMDetection 3.3.0**: Used for the Co-DERT and RTMDet models.
- **Ultralytics (latest version)**: Used for YOLOv10.
- **TensorRT**: For model export and inference acceleration.
- Other required libraries can be found in the `requirements.txt`.

## Tools Overview

The `tools` directory contains the following scripts:

- **crop_image.py**: Crops images based on specified coordinates and generates the corresponding annotations.
- **export_tensorrt.py**: Converts PyTorch `.pt` weight files to TensorRT `.engine` files, performs inference, and saves the results to a file.
- **inference_script.py**: Runs inference on input images and outputs the results to a file.
- **txt2json.py**: Converts annotation files from TXT format to JSON.
- **voc2txt.py**: Converts VOC format annotations to TXT.

## Usage Instructions

### Inference

Please download pre-trained weights first.

To get dense outputs of `Co-DERT` and `RTMDet`, please use following scripts in corresponding foler:

```bash
python infer.py \
    --config <config_path> \
    --ckpt <ckpt_path> \
    --img_dir <image_dir> \
    --out_dir <output_dir>
```

Edit infer.py for more configuration settings.

For the final results, please use the `inference_script.py` to perform inference on a set of images:

```bash
python tools/inference_script.py --model-file <model weight, including pt, onnx, engine > --source-path <path_to_images> --output-file <path_to_output> --engine < whether export the pt to engine for inference >
```

### Export to TensorRT

To export a PyTorch model to TensorRT and run inference:

```bash
python tools/export_tensorrt.py --model-file <model weight, including pth, onnx, engine > --source-path <path_to_images> --output-file <path_to_output>
```

### Training

To train models using MMDetection:

1. Navigate to the `Co-DERT` or `RTMDet` folder.
2. Modify the configuration files as needed.
3. Run the training script:
   ```bash
   sh dist_train.sh configs/<your_config>.py GPU_NUM --work-dir <your_work_dir>
   ```

For YOLOv10, training instructions can be found in the `YOLOv10/README.md` file.

## Acknowledgements

This project is based on the following repositories:
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [Road Damage Detector](https://github.com/sekilab/RoadDamageDetector)
