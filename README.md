# Income Prediction

Income Prediction is a machine learning project that predicts whether an individual's income is greater than or less than $50K based on various features. This project includes data preprocessing, exploratory data analysis, model training, and deployment using Flask.

---

## Project Structure
Income Prediction/ │ ├── app.py # Flask application for model deployment ├── requirements.txt # Python dependencies ├── setup.py # Project setup configuration ├── .gitignore # Files and directories to ignore in version control ├── README.md # Project documentation │ ├── artifacts/ # Directory for storing model and preprocessor artifacts │ ├── model.pkl # Trained machine learning model │ ├── preprocessor.pkl # Preprocessing pipeline │ ├── raw.csv # Raw dataset │ ├── train.csv # Processed training dataset │ └── test.csv # Processed testing dataset │ ├── logs/ # Directory for application logs │ ├── <timestamp>.log # Log files for debugging and monitoring │ ├── notebooks/ # Jupyter notebooks for EDA and model training │ ├── EDA.ipynb # Exploratory Data Analysis notebook │ └── Model Training.ipynb # Model training and evaluation notebook │ ├── src/ # Source code for the project │ ├── init.py # Package initialization │ ├── exception.py # Custom exception handling │ ├── logger.py # Logging utility │ ├── utils.py # Utility functions │ ├── components/ # Components for data processing and modeling │ └── pipelines/ # Pipelines for end-to-end workflows │ ├── templates/ # HTML templates for the Flask web app │ └── index.html # Frontend form for user input │ └── .gitignore # Specifies files and directories to ignore in Git

---

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
- **Exploratory Data Analysis (EDA)**: Provides insights into the dataset using visualizations and statistical analysis.
- **Model Training**: Trains machine learning models like Naive Bayes, Decision Trees, or XGBoost to predict income.
- **Model Deployment**: Deploys the trained model using Flask for real-time predictions.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Income-Prediction
2. Install dependencies:
    pip install -r requirements.txt
3. Run the Flask application:
    python app.py
4. Open your browser and navigate to http://127.0.0.1:5000/ to access the web interface.

## Dataset
The dataset used for this project contains features like age, workclass, education, marital status, occupation, etc., to predict income. The data is split into training and testing sets for model evaluation.

## Logs
Logs are stored in the logs/ directory to track the application's behavior and debug issues.

## Artifacts
model.pkl: Trained machine learning model.
preprocessor.pkl: Preprocessing pipeline for input data.
raw.csv: Raw dataset.
train.csv and test.csv: Processed training and testing datasets.

## Frontend
The frontend is a simple HTML form (templates/index.html) where users can input data for prediction.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Contact
For any questions or feedback, feel free to reach out.


