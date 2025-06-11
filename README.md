# Telco Churn Prediction Project

This project implements a machine learning model to predict customer churn for a telecommunications company. It includes a Streamlit web application for interactive predictions, a Telegram bot for chat-based predictions, and a Jupyter notebook for model training and evaluation.

## Project Structure

```
FINAL_PROJECT/
├── bot/
│   └── bot.py                          # Telegram bot implementation
├── data/
│   └── data.csv                       # Original datasetgit rm --cached data/cleaned_telco_data.csv
git commit -m "Remove large CSV file from repo"
git push origin main
├── models/
│   ├── best_churn_model.pkl           # Trained Random Forest model
│   ├── cat_columns.pkl                # Categorical feature columns
│   ├── num_features.pkl               # Numerical features list
│   └── scaler.pkl                     # StandardScaler for preprocessing
├── web/
│   └── app.py                         # Streamlit web application
├── question.pdf                       # Project requirements
├── telecom_churn_modeling.ipynb       # Model training notebook
└── README.md                          # This file
```

## Components

### 1. Model Training (`telecom_churn_modeling.ipynb`)
- Jupyter notebook for data preprocessing and model training
- Trains a Random Forest classifier for churn prediction
- Saves trained model and preprocessing artifacts to `models/` directory
- **Outputs:**
  - `best_churn_model.pkl` - Trained Random Forest model
  - `scaler.pkl` - StandardScaler for numerical features
  - `cat_columns.pkl` - List of categorical feature columns
  - `num_features.pkl` - List of numerical features

### 2. Telegram Bot (`bot/bot.py`)
- Interactive Telegram bot for churn predictions
- Users input customer data through chat interface
- Returns churn probability predictions
- Uses saved model and preprocessing artifacts

### 3. Web Application (`web/app.py`)
- Streamlit-based web interface
- User-friendly form with 19 input fields
- Real-time churn probability predictions
- Responsive design for desktop and mobile

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser (for Streamlit app)
- Telegram account (for bot usage)
- Jupyter Notebook (for model training)

## Installation & Setup

### 1. Clone and Navigate to Project Directory
```bash
cd FINAL_PROJECT/
```

### 2. Install Required Dependencies
```bash
pip install streamlit pandas numpy scikit-learn python-telegram-bot jupyter
```

**Alternative: Using Virtual Environment (Recommended)**
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install streamlit pandas numpy scikit-learn python-telegram-bot jupyter
```

### 3. Verify Model Files
Ensure all required model files exist in the `models/` directory:
```bash
# Windows
dir models

# macOS/Linux
ls models/
```

Required files:
- `best_churn_model.pkl`
- `scaler.pkl`
- `cat_columns.pkl`
- `num_features.pkl`

### 4. Validate Model Files (Optional)
Verify that `scaler.pkl` contains a proper StandardScaler object:
```bash
python -c "import pickle; print(type(pickle.load(open('models/scaler.pkl', 'rb'))))"
```
Expected output: `<class 'sklearn.preprocessing._data.StandardScaler'>`

## Usage Instructions

### Running the Web Application

1. **Start the Streamlit server:**
   ```bash
   streamlit run web/app.py
   ```

2. **Access the application:**
   - Browser should automatically open to `http://localhost:8501`
   - If not, manually navigate to `http://localhost:8501`

3. **If port 8501 is busy:**
   ```bash
   streamlit run web/app.py --server.port 8502
   ```

4. **Using the web interface:**
   - Fill out all 19 form fields (customer information)
   - Ensure all dropdown menus have selections
   - Enter valid numerical values (non-negative for tenure, charges)
   - Click "Predict Churn" to get probability percentage

### Running the Telegram Bot

1. **Get Telegram Bot Token:**
   - Message @BotFather on Telegram
   - Create new bot and obtain token

2. **Configure the bot:**
   - Open `bot/bot.py`
   - Replace placeholder with your bot token

3. **Start the bot:**
   ```bash
   python bot/bot.py
   ```

4. **Interact with bot:**
   - Find your bot on Telegram
   - Send customer data as requested
   - Receive churn probability predictions

### Model Training (If Needed)

If model files are missing or corrupted:

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open and run notebook:**
   - Open `telecom_churn_modeling.ipynb`
   - Ensure `data/cleaned_telco_data.csv` exists
   - Run all cells to generate new model files

3. **Verify dataset:**
   ```bash
   # Windows
   dir data
   # macOS/Linux
   ls data/
   ```

## Data Requirements

The dataset (`data/cleaned_telco_data.csv`) should contain the following features:
- **Numerical:** tenure, MonthlyCharges, TotalCharges
- **Categorical:** gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
- **Target:** Churn (Yes/No)

## Troubleshooting

### Model Loading Issues
```bash
# Check if all model files exist
dir models  # Windows
ls models/  # macOS/Linux

# Verify scaler type
python -c "import pickle; print(type(pickle.load(open('models/scaler.pkl', 'rb'))))"
```

### Streamlit Issues
```bash
# Verify Streamlit installation
pip show streamlit

# Run from correct directory
cd FINAL_PROJECT/
streamlit run web/app.py
```

### Form Validation Errors
- Ensure all dropdown fields have selections
- Verify numerical inputs are valid (non-negative values)
- Check feature lists:
```bash
python -c "import pickle; print(pickle.load(open('models/cat_columns.pkl', 'rb')))"
python -c "import pickle; print(pickle.load(open('models/num_features.pkl', 'rb')))"
```

### Bot Issues
- Verify bot token is correctly set in `bot.py`
- Ensure bot script is running continuously
- Check network connectivity

## Technical Details

- **Model:** Random Forest Classifier
- **Framework:** Streamlit for web interface
- **Bot Platform:** Telegram Bot API
- **Data Processing:** pandas, scikit-learn
- **Deployment:** Local server (expandable to cloud)

## Performance

The model provides churn probability predictions with training accuracy metrics available in the Jupyter notebook. Results are displayed as percentages (e.g., "Sizning ketish ehtimolingiz: 78%").

## Future Enhancements

- Cloud deployment (AWS, Heroku, etc.)
- Database integration for prediction logging
- Advanced model ensemble methods
- Real-time model retraining capabilities
- Additional communication channels (WhatsApp, Slack)

## Support

For detailed project requirements and specifications, refer to `question.pdf`. For code-specific questions, check the comments in individual files (`app.py`, `bot.py`, `telecom_churn_modeling.ipynb`).