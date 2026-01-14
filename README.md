# ML-assignment-2---Bits
# Problem statement
Implementation and deployment of six machine learning classification models with evaluation using Streamlit.
use Logestic regression, Deciosion Tree, KNN, Naive Bayes, Random forest, XGB for a selected binary calssification problem, using a selected datset.

# Dataset description 
Dataset : UCI Heart Disease Data

Dataset link : https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

Features : 14 (12 features , 1 target, 1 id)

sample size : 920


| Model               | Accuracy | AUC     | Precision | Recall  | F1      | MCC     |
|---------------------|----------|---------|-----------|---------|---------|---------|
| Logistic Regression | 0.8261   | 0.9118  | 0.8241    | 0.8725  | 0.8476  | 0.6469  |
| KNN                 | 0.8424   | 0.8897  | 0.8288    | 0.9020  | 0.8638  | 0.6810  |
| Naive Bayes         | 0.8533   | 0.9157  | 0.8505    | 0.8922  | 0.8708  | 0.7023  |
| Decision Tree       | 0.7391   | 0.7348  | 0.7596    | 0.7745  | 0.7670  | 0.4709  |
| Random Forest       | 0.8587   | 0.9179  | 0.8725    | 0.8725  | 0.8725  | 0.7140  |
| XGBoost             | 0.8424   | 0.8931  | 0.8476    | 0.8725  | 0.8599  | 0.6802  |




| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | shows strong and balanced performance with good accuracy and high Aauc, indicating good class separability. Precision and recall are well balanced, making it a reliable and interpretable baseline model for heart disease prediction task. This can be considered as one of the simple models. |
| Decision tree | Performs comparatively weaker than other models, with lower accuracy, AUC, and MCC, suggesting overfitting and limited generalization on unseen data. |
| KNN | Achieves good recall and F1-score, indicating effective identification of positive heart disease cases. However, slightly lower AUC suggests moderate discrimination capability |
| Naive Bayes | Deliver consistent and competitive performance with high AUC and balanced precision and recall, working well despite its strong feature independence assumption. |
| Random Forest (Ensemble) | One of the best-performing models overall, achieving the highest accuracy, auc, F1score, and mcc. Ensemble learning helps reduce overfitting and capture complex patterns. |
| XGBoost (Ensemble) | Shows strong and stable performance with high accuracy and F1score, slightly below Random Forest, and effectively models non-linear relationships in the dataset |

