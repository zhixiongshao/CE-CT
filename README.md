# CE-CT
This is a model to predict perivascular invasion in pancreatic ductal adenocarcinoma

### 1. Tumor Segmentation using nnU-Net

- The CT images are first processed using [nnU-Net](https://github.com/MIC-DKFZ/nnunet), a self-configuring deep learning framework for medical image segmentation.  
- nnU-Net automatically adapts preprocessing, network architecture, and training pipelines to your dataset.  
- In this project, nnU-Net is used to segment the pancreatic tumor regions from CT scans.  
- For more details, see the original repository: [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnunet)

### 2. Perivascular Invasion Prediction

- After segmentation, the tumor regions are used as input for a prediction model that estimates the likelihood of vascular invasion.  
- The prediction model is trained using labeled cases with and without vascular invasion.  
- Clinical data can be integrated as additional input features for multi-modal prediction.

### 3. Model Weights and Test Data

- Pre-trained model weights are provided in the `model/` directory.  
- Example test data are available in the `test_predict/` and `test_segment/` directory for evaluation and reproducibility.  

---


