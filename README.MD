## Link for Datasets and other files:
# https://drive.google.com/drive/folders/1a1ej6f_MyhsjI51LtIXsWknMruYr869N?usp=sharing


Traffic Analysis with YOLOv11
This project provides a setup for performing traffic analysis using a pretrained YOLOv11 model. Follow the instructions below to set up the project, install dependencies, and run the analysis on a traffic video.


Project Structure
Ensure the following files and folders are in the same root directory:
root/
├── train/                   # Training dataset
├── test/                    # Test dataset
├── valid/                   # Validation dataset
├── data.yaml                # Dataset configuration file
├── Traffic_Analysis.ipynb   # Jupyter notebook for traffic analysis
├── yolo11s.pt               # Pretrained YOLOv11 small model weights


Download YOLOv11 Weights
Place your pretrained YOLOv11 weights file (e.g., yolo11s.pt) in the root directory, as shown in the project structure. 

Open the Traffic_Analysis.ipynb Jupyter notebook.
Run the cells one by one to execute the traffic analysis pipeline.
Ensure all paths in the notebook (e.g., to datasets, weights, or videos) are correctly set to match your local or Colab environment.

Notes for Google Colab
If running this project in Google Colab:

Connect to Google Drive:
Mount your Google Drive to access the project files by including the following code in the first cell of the notebook:from google.colab import drive
drive.mount('/content/drive')


Update file paths in the notebook to point to your Google Drive directory (e.g., /content/drive/MyDrive/root/).


Ensure the project structure (as shown above) is replicated in your Google Drive.
Verify that the yolo11s.pt weights file and data.yaml are accessible in the specified paths.

Uploading and Downloading Videos

Uploading a Traffic Video:
You can upload a traffic video to analyze in the Jupyter notebook. If using Colab, use the file upload feature:from google.colab import files
uploaded = files.upload()


Update the notebook to reference the uploaded video file's path.


Downloading the Output:
Running the last cell of notebook will provide an option to download the analyzed video 
In Colab, use the following code to download the output video:from google.colab import files
files.download('path/to/output_video.mp4')


Replace 'path/to/output_video.mp4' with the actual path to your processed video.



Additional Notes

Ensure all dependencies are installed correctly before running the notebook.
If you encounter path-related errors, double-check the file paths in data.yaml and the notebook.
For large datasets or videos, ensure sufficient storage and memory, especially when using Colab.
