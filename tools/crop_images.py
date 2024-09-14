import cv2
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


def cal_anno_size(image_paths, annotations_paths):

    xxmin,xxmax,yymin,yymax = 0,0,0,0
    odd_count = 0
    boxes =[]
    for idx,(image_path,annotations_path) in enumerate(zip(image_paths, annotations_paths)):
        # Load image
        image = Image.open(image_path)

        # Load annotations
        with open(annotations_path, 'r') as file:
            annotations = file.readlines()

        # Update annotations
        for line in annotations:
            parts = line.strip().split()
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])


            original_width, original_height = image.size

            xmin = (x - w /2 )* original_width
            xmax = (x + w/2 )* original_width
            ymin = (y -h /2 )* original_height
            ymax = (y + h/2 )* original_height

            boxes.append((xmin, xmax, ymin, ymax))

            if xmax > 3000:
                odd_count += 1

            if xmin < 0 or ymin < 0:
                continue

            if idx == 0:
                xxmin = xmin
                xxmax = xmax
                yymin = ymin
                yymax = ymax

            xxmin = min(xxmin,xmin)
            xxmax = max(xxmax,xmax)
            yymin = min(yymin,ymin)
            yymax = max(yymax,ymax)

    print(f"xmin:{xxmin},xmax:{xxmax},ymin{yymin},ymax{yymax}")
    print(f'odd count {odd_count}')
    plot_bbox_distribution(boxes)

def crop_annotations(image_path,annotation_path):
    # Load image
    image = Image.open(image_path)
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()
    if len(annotations) == 0:
        return

    for idx,line in enumerate(annotations):
        parts = line.strip().split()
        cls = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])


        original_width, original_height = image.size

        xmin = int((x - w /2 )* original_width)
        xmax = int((x + w/2 )* original_width)
        ymin = int((y -h /2 )* original_height)
        ymax = int((y + h/2 )* original_height)

        crop = image.crop((xmin, ymin, xmax, ymax))
        save_path =str(image_path.parent).replace('images','crop_annotations') + '/' + str(image_path.stem)
        os.makedirs(save_path,exist_ok=True)
        crop.save(save_path + '/' +f'{idx}.png' )

def crop_image_with_yolo_annotations(image_path, annotations_path , crop_box, output_image_path,
                                     output_annotations_path):
    """
    :param image_path: Path to the original image.
    :param annotations_path: Path to the original annotations (in YOLO format).
    :param crop_box: A tuple (left, upper, right, lower) defining the crop box.
    :param output_image_path: Path to save the cropped image.
    :param output_annotations_path: Path to save the updated annotations.
    """
    # Load image
    image = Image.open(image_path)

    # Crop image
    cropped_image = image.crop(crop_box)
    cropped_image.save(output_image_path)

    # Load annotations
    if annotations_path is not None:
        with open(annotations_path, 'r') as file:
            annotations = file.readlines()

        # Update annotations
        updated_annotations = ''
        for line in annotations:
            parts = line.strip().split()
            cls = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])


            original_width, original_height = image.size
            left, upper, right, lower = crop_box

            xmin = (x - w /2 )* original_width
            xmax = (x + w/2 )* original_width
            ymin = (y -h /2 )* original_height
            ymax = (y + h/2 )* original_height

            if (left <= x * original_width <= right) and (upper <= y * original_height <= lower):
                new_x = (x * original_width - left) / (right - left)
                new_y = (y * original_height - upper) / (lower - upper)
                new_w = w * original_width / (right - left)
                new_h = h * original_height / (lower - upper)

                # updated_annotations.append({
                #     'cls': cls,
                #     'x': new_x,
                #     'y': new_y,
                #     'w': new_w,
                #     'h': new_h
                # })
                updated_annotations += f'{cls} {new_x} {new_y} {new_w} {new_h}\n'

        # Save updated annotations
        with open(output_annotations_path, 'w') as file:
            file.write(updated_annotations)

    # Save the relative relationship between the original and cropped coordinates
    # relationship = {
    #     'crop_box': crop_box,
    #     'original_annotations': annotations,
    #     'cropped_annotations': updated_annotations
    # }

    # with open(output_annotations_path.replace('.txt', '_relationship.json'), 'w') as file:
    #     json.dump(relationship, file)


def plot_bbox_distribution(boxes):
    """
    Plot the distribution of bounding box edges.
    """
    xmin_coords = [box[0] for box in boxes]
    xmax_coords = [box[1] for box in boxes]
    ymin_coords = [box[2] for box in boxes]
    ymax_coords = [box[3] for box in boxes]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(xmin_coords, ymin_coords, alpha=0.5, label='xmin, ymin')
    plt.scatter(xmax_coords, ymax_coords, alpha=0.5, label='xmax, ymax')
    plt.title('Distribution of Bounding Box Edges')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(xmin_coords, xmax_coords, alpha=0.5, label='xmin, xmax')
    plt.scatter(ymin_coords, ymax_coords, alpha=0.5, label='ymin, ymax')
    plt.title('Distribution of Bounding Box Edges')
    plt.xlabel('Min Coordinate')
    plt.ylabel('Max Coordinate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.savefig(output_image_path)
    plt.show()

# Example usage

image_paths = list(Path(r"/data0/weijn/RDD2024/Norway/test/images").glob('*.jpg'))
annotations_paths = [None] * len(image_paths ) # YOLO format file
os.makedirs(r"/data0/weijn/RDD2024/Norway/test/crop/images",exist_ok=True)

crop_box = (0, 1000, 2500, 2035)  # Example crop box

for image_path,annotations_path in zip(image_paths,annotations_paths):
    # pass
    crop_image_with_yolo_annotations(image_path, annotations_path, crop_box, str(image_path).replace('images',r'crop/images'), str(annotations_path).replace('labels','crop_labels'))
    # crop_annotations(image_path,annotations_path)