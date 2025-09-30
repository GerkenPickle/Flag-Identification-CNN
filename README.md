# Flag Classification CNN Benchmark

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify country flags. The goal is to evaluate the performance of CPU and GPU hardware in running the model by measuring metrics such as runtime, throughput, and efficiency. Data augmentation techniques were applied to improve model robustness under various conditions.

## Files Included
- [`Flag_Classification_CNN_Benchmark.ipynb`](https://github.com/GerkenPickle/Flag-Identification-CNN/blob/main/CNN_Flag_Classifier.ipynb) — Google Colab notebook containing all code for data preprocessing, model training, and CPU/GPU benchmarking.
- [`Project_Description.pdf`](https://github.com/GerkenPickle/Flag-Identification-CNN/blob/main/Project_Description.pdf) — A PDF describing the project goals and methodology.
- [`Final_Report.pdf`](https://github.com/GerkenPickle/Flag-Identification-CNN/blob/main/Final_Report.pdf) — A PDF containing full experimental results, methods, and bibliographic study.

## Requirements
- Python 3.x  
- PyTorch  
- torchvision  
- fvcore  
- psutil  
- PIL / Pillow  

All packages can be installed in Colab or via pip.

## Setup and Usage
0. (Optional) You can open and run the notebook directly in Google Colab without downloading anything:  
[Open in Colab](https://colab.research.google.com/drive/1TjjmaYBPVGBHLphZaSEiiSN-H6qtnJ33?usp=sharing)


      **OR**

1. Open the notebook in [Google Colab](https://colab.research.google.com/).  
2. Run the notebook cells in order. The code will automatically download the dataset and test images.  
3. The notebook will train, benchmark on CPU and GPU, and display results.  

## Key Features
- CNN architecture for flag classification  
- Data augmentation using ImageNet AutoAugment policies  
- Benchmarking of CPU vs GPU runtime, throughput, and efficiency  
- Visualization of performance metrics  

## Results
The notebook reports:
- Accuracy (%) on test dataset  
- Runtime (seconds) for CPU and GPU  
- Throughput (GFLOPs)  
- Efficiency (GFLOPs/W for GPU)  

## Notes
- All code is designed to run in Google Colab.
- Dataset and test images are downloaded dynamically; no additional uploads are required.
- **Data Set Source:** [Countries Flags Images on Kaggle](https://www.kaggle.com/datasets/yusufyldz/countries-flags-images)
