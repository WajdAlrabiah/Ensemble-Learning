# Ensemble Learning

This lab demonstrates three fundamental **ensemble learning techniques** — Bagging, Random Forest, and AdaBoost — using the **moon-shaped dataset**.  
Each model combines multiple weak learners (Decision Trees) to improve prediction accuracy and robustness.

---

## Project Overview
This notebook contains three main parts:

### Problem 1: Bagging Classifier
- Uses `BaggingClassifier` with `DecisionTreeClassifier` as base estimators  
- Combines 500 trees trained on random subsets of the data  
- Visualizes decision boundaries  
- Calculates model accuracy  

### Problem 2: Random Forest
- Implements `RandomForestClassifier` with 500 trees and `max_leaf_nodes=16`  
- Automatically introduces randomness in feature selection  
- Evaluates performance on the test set  

### Problem 3: AdaBoost
- Uses `AdaBoostClassifier` with 200 weak learners (`DecisionTreeClassifier(max_depth=1)`)  
- Sequentially focuses on misclassified samples  
- Shows improved performance through adaptive weighting  

---

## Key Concepts
- **Bagging (Bootstrap Aggregating):** Reduces variance by training models on random subsets  
- **Random Forest:** Introduces feature randomness to decorrelate trees  
- **AdaBoost:** Sequentially reweights data to correct weak learner errors  
- **Decision Boundary Visualization:** Understands how ensemble methods partition feature space  

---

## Libraries Used
- `numpy`
- `matplotlib`
- `scikit-learn`

---
