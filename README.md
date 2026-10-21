# 🏠 House Price Prediction

A Machine Learning web application that estimates house prices based on property features using Python, Scikit-learn, and Streamlit.

## ✨ Features

- 🏡 Predict estimated house prices using property details
- 📐 Input built-up area, bedrooms, bathrooms, and property age
- 📊 Interactive data visualizations using Plotly
- 🌙 Modern dark-themed user interface
- 📈 Linear Regression model for price estimation
- 📋 View the sample dataset directly in the app

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, Linear Regression
- **Data Processing:** Pandas
- **Web Framework:** Streamlit
- **Visualization:** Plotly
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
house-price-prediction/
├── app.py
├── train_model.py
├── house_data.csv
├── house_price_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/snehakumari1984/house-price-prediction.git
cd house-price-prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional if the saved model exists)

```bash
python train_model.py
```

### 5. Run the application

```bash
streamlit run app.py
```

## 🧠 How It Works

1. Loads the housing dataset from a CSV file.
2. Uses area, bedrooms, bathrooms, and property age as input features.
3. Trains a Linear Regression model on the sample dataset.
4. Evaluates the model using Mean Absolute Error (MAE) and R² score.
5. Saves the trained model and uses it to estimate prices for new property inputs.
6. Displays the estimate and interactive visualizations in Streamlit.

## 📊 Model Evaluation

The current training run reported:

- **Mean Absolute Error (MAE):** 9.38 lakhs
- **R² Score:** 0.97

These metrics come from a small synthetic dataset and a single train/test split. They do not establish real-world house-price accuracy.

## ⚠️ Limitations

- Uses a small, synthetic sample dataset rather than verified real estate records.
- Does not currently consider city, locality, land value, or market trends.
- Predictions are educational estimates and should not be used for actual property buying or selling decisions.

## 🚀 Future Improvements

- Use a larger, real-world housing dataset.
- Add city and locality as features.
- Compare multiple regression algorithms.
- Add model validation and prediction error visualizations.
- Deploy the application online.


## 🚀 Live Demo

[Click here to try the app](


## 👩‍💻 Author

**Sneha Kumari**

GitHub: [snehakumari1984](https://github.com/snehakumari1984)

---

*Built as a hands-on Machine Learning project to practice regression, data processing, visualization, and web application development.*
