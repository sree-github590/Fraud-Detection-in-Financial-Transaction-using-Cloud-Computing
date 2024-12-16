# Fraud-Detection-in-Financial-Transaction-using-Cloud-Computing

This project implements a **fraud detection system** for financial transactions, combining **machine learning (ML)** with **AWS EC2 cloud infrastructure** for real-time predictions. The system is built upon a modified version of a pre-trained ML model from [RaimbekovA's repository](https://github.com/RaimbekovA/bank-card-fraud-detection-using-machine-learning.git), with customizations made to fit our use case. The training dataset is sourced from Kaggle.  

---

## Overview  
Fraud detection is essential to ensure financial security. This project integrates a cloud-based deployment with a customized ML model to detect fraudulent transactions effectively and efficiently.  

### Key Features  
1. **Machine Learning**:  
   - Adapted one model (Decision Trees) from the original repository.  
   - Modified the preprocessing steps and fine-tuned the hyperparameters for better performance.  
   - Dataset: Used Kaggle's credit card fraud detection dataset, which includes 284,807 transactions.  
2. **AWS EC2 Deployment**:  
   - Scalable and reliable hosting of the application.  
3. **RESTful APIs**: Enables easy integration into other systems for fraud detection.  
4. **User-Friendly Interface**: Streamlit-based application for manual input and predictions.  

---

## Architecture  

### Workflow  
1. **Data Input**:  
   - Transaction data is submitted through APIs or a web interface.  
2. **Data Preprocessing**:  
   - The input data is standardized and cleaned based on Kaggle's dataset schema.  
3. **ML Model Inference**:  
   - The Logistic Regression model predicts whether the transaction is fraudulent.  
4. **Result Output**:  
   - Results are displayed via the API or interface, marked as "Fraudulent" or "Non-Fraudulent."  
5. **Cloud Hosting**:  
   - Deployed on AWS EC2 for real-time and scalable processing.  

---

## Technologies Used  

- **AWS EC2**: Cloud hosting and scaling.  
- **Python**: Backend and ML integration.  
- **Streamlit**: Web interface for manual verification.  
- **Scikit-learn**: ML framework for training and inference.  
- **Kaggle Dataset**: Source of transaction data for model training and validation.  

---

## Getting Started  

### Prerequisites  
1. **AWS EC2 Instance**:  
   - Launch an Ubuntu-based EC2 instance and configure security groups for required ports.  
2. **Python Environment**: Ensure Python 3.8+ is installed with pip.  

### Installation  
1. Clone this repository and download the dataset from Kaggle:  
   ```bash  
   git clone https://github.com/your-repo/fraud-detection-using-cloud  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Upload the project to the EC2 instance using SCP or AWS CLI.  

### Running the Application  
1. SSH into your EC2 instance.  
2. Start the application:  
   ```bash  
   python app.py  
   ```  
3. Access the application or API via your EC2 public IP.  

---

## Example Usage  

### API Request  
Send a POST request to the API endpoint:  
```bash  
curl -X POST http://<EC2_PUBLIC_IP>:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"transaction_amount": 500, "transaction_time": 2, "card_type": "debit"}'  
```  

### Streamlit Interface  
Access the web application at:  
```plaintext  
http://<EC2_PUBLIC_IP>:8501/  
```  


## Repository Structure  
```plaintext  
.  
├── app.py               # Main application logic  
├── model/               # Customized ML model files  
├── requirements.txt     # Python dependencies  
├── README.md            # Documentation  
├── utils/               # Helper scripts  
└── data/                # Kaggle dataset  
```  

---

## Customizations and Contributions  
- Modified the model from the original [RaimbekovA repository](https://github.com/RaimbekovA/bank-card-fraud-detection-using-machine-learning.git).  
- Integrated custom preprocessing pipelines to better align with the Kaggle dataset.  
- Deployed an optimized API for faster predictions.  

---

## References  
- **Kaggle Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Original ML Model Repository**: [RaimbekovA's GitHub](https://github.com/RaimbekovA/bank-card-fraud-detection-using-machine-learning.git)  

--- 

Feel free to contribute or open issues for discussion!
