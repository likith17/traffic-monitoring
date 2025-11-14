# Traffic Analysis with YOLOv11

This project provides a full setup for performing traffic analysis using a pretrained YOLOv11 model. It supports vehicle detection, congestion estimation, and video processing for both local environments and Google Colab.


---
## Project Structure

Ensure the following files and directories exist in the project root:

```text
root/
├── train/                   # Training dataset
├── test/                    # Test dataset
├── valid/                   # Validation dataset
├── data.yaml                # Dataset configuration file
├── Traffic_Analysis.ipynb   # Jupyter notebook for traffic analysis
└── yolo11s.pt               # Pretrained YOLOv11 model weights
```

Replicate the same structure if working from Google Drive or Google Colab.

---

## Setting Up YOLOv11 Weights

1. Download your pretrained YOLOv11 weights file (e.g., `yolo11s.pt`).
2. Place the weights file inside the root directory.
3. Open the `Traffic_Analysis.ipynb` notebook.
4. Confirm that all paths inside the notebook match your environment.

---

## Running the Notebook

1. Open the `Traffic_Analysis.ipynb` notebook.
2. Execute the cells in order.
3. Ensure the following paths point to valid locations:
   - Dataset folders (`train/`, `test/`, `valid/`)
   - YOLOv11 weights (`yolo11s.pt`)
   - `data.yaml`
   - Input traffic video file

The notebook will:
- Load the dataset
- Load the YOLOv11 model
- Run inference on traffic video frames
- Annotate the detections
- Save the processed output video

---

