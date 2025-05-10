APP INSTALLATION

!pip install gradio
!pip install scikit-learn
!pip install matplotlib seaborn
!pip install --upgrade gradio scikit-learn pandas

UPLOAD THE DATASET

from google.colab import files

uploaded = files.upload()  # Upload 'accident.csv'

LOAD THE DATASET

import pandas as pd

url = 'https://raw.githubusercontent.com/adduadnanali/Road-Accident-Analysis-in-India/main/accident.csv'
df = pd.read_csv(url)

# Show first few rows
df.head()

DATA EXPLORATION

# Check basic information about the dataset
df.info()

# Check for any missing values
df.isnull().sum()

# Check for duplicates
df.duplicated().sum()

# Get summary statistics for numerical features
df.describe()

# Check missing values
df.isnull().sum()

# Drop duplicates if any
df.drop_duplicates(inplace=True)

VISUALIZATION

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='State', order=df['State'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number of Accidents per State')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(df['Number_of_Deaths'], bins=10, kde=True)
plt.title('Distribution of Deaths per Accident')
plt.xlabel('Number of Deaths')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Weather_Conditions', order=df['Weather_Conditions'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Accidents by Weather Conditions')
plt.tight_layout()
plt.show()

# Convert death and injury counts to integers
df['Number_of_Deaths'] = pd.to_numeric(df['Number_of_Deaths'], errors='coerce')
df['Number_of_Injuries'] = pd.to_numeric(df['Number_of_Injuries'], errors='coerce')

# Fill any NaNs (caused by conversion errors) with 0
df['Number_of_Deaths'] = df['Number_of_Deaths'].fillna(0).astype(int)
df['Number_of_Injuries'] = df['Number_of_Injuries'].fillna(0).astype(int)

# Now apply the function to create Risk_Level
def classify_risk(row):
    total = row['Number_of_Deaths'] + row['Number_of_Injuries']
    if total >= 5:
        return 'High'
    elif total >= 2:
        return 'Medium'
    else:
        return 'Low'

df['Risk_Level'] = df.apply(classify_risk, axis=1)

# First, replace any non-numeric values (like 'Unknown') with 0
df['Number_of_Deaths'] = pd.to_numeric(df['Number_of_Deaths'], errors='coerce').fillna(0).astype(int)
df['Number_of_Injuries'] = pd.to_numeric(df['Number_of_Injuries'], errors='coerce').fillna(0).astype(int)
def classify_risk(row):
    total = row['Number_of_Deaths'] + row['Number_of_Injuries']
    if total >= 5:
        return 'High'
    elif total >= 2:
        return 'Medium'
    else:
        return 'Low'

df['Risk_Level'] = df.apply(classify_risk, axis=1)
df[['Number_of_Deaths', 'Number_of_Injuries', 'Risk_Level']].head()

# Select useful features + the target column
df_model = df[['Weather_Conditions', 'Road_Type', 'Road_Conditions',
               'Alcohol_Involved', 'Driver_Fatigue', 'Speed_Limit', 'Time', 'Risk_Level']].copy()

# Convert 'Time' to Hour
df_model['Hour'] = pd.to_datetime(df_model['Time'], errors='coerce').dt.hour.fillna(0).astype(int)

# Drop the original 'Time' column now that we've extracted hour
df_model.drop('Time', axis=1, inplace=True)

# Convert categorical features using one-hot encoding
df_encoded = pd.get_dummies(df_model, columns=['Weather_Conditions', 'Road_Type',
                                               'Road_Conditions', 'Alcohol_Involved',
                                               'Driver_Fatigue'], drop_first=True)

# Separate features and target
X = df_encoded.drop('Risk_Level', axis=1)
y = df_encoded['Risk_Level']

from sklearn.preprocessing import LabelEncoder

# Encode the target labels (Risk_Level)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

def predict_risk(weather, road_type, road_condition, alcohol, fatigue, speed_limit, hour):
    # Debug: Check what inputs are being passed
    print(f"Weather: {weather}, Road Type: {road_type}, Road Condition: {road_condition}, Alcohol: {alcohol}, Fatigue: {fatigue}, Speed Limit: {speed_limit}, Hour: {hour}")
    
    # Create a DataFrame from input
    input_data = pd.DataFrame([{
        'Speed_Limit': float(speed_limit),
        'Hour': int(hour),
        'Weather_Conditions_' + weather: 1,
        'Road_Type_' + road_type: 1,
        'Road_Conditions_' + road_condition: 1,
        'Alcohol_Involved_' + alcohol: 1,
        'Driver_Fatigue_' + fatigue: 1
    }])

    # Add missing columns with 0
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns
    input_data = input_data[X.columns]

    # Scale the features
    input_scaled = scaler.transform(input_data)

    # Predict
    prob = model.predict_proba(input_scaled)[0]
    pred = model.predict(input_scaled)[0]

    risk_label = le.inverse_transform([pred])[0]
    confidence = max(prob)

    # Suggested action (custom logic)
    if risk_label == 'High':
        suggestion = "⚠️ Slow down, avoid peak hours, check road alerts."
    elif risk_label == 'Medium':
        suggestion = "🟡 Drive cautiously, maintain safe speed."
    else:
        suggestion = "🟢 Safe to proceed, but stay alert."

    return risk_label, f"{confidence*100:.2f}%", suggestion

import gradio as gr
import numpy as np
import pandas as pd

# Assuming X is a DataFrame with your model features
# This should be defined before this code or loaded from somewhere
# For demonstration, I'll add a mock predict_risk function

def predict_risk(weather, road_type, road_cond, alcohol, fatigue, speed, hour):
    """
    Predicts traffic risk based on input parameters.
    
    Args:
        weather: Selected weather condition
        road_type: Selected road type
        road_cond: Selected road condition
        alcohol: Whether alcohol is involved
        fatigue: Whether driver fatigue is involved
        speed: Speed limit in km/h
        hour: Hour of the day (0-23)
        
    Returns:
        Tuple of (risk level, probability, suggested action)
    """
    # Create a feature array for prediction
    # In a real implementation, you would:
    # 1. Create a zero-vector matching your model's expected input size
    # 2. Set the appropriate one-hot encoded features to 1
    # 3. Run the vector through your prediction model
    
    # For this example, I'll use a simple rule-based approach
    risk_score = 0
    
    # Weather factors
    high_risk_weather = ["Rain", "Snow", "Fog", "Storm"]
    if weather in high_risk_weather:
        risk_score += 30
    
    # Road type factors
    if road_type in ["Highway", "Rural"]:
        risk_score += 20
    
    # Road condition factors
    if road_cond in ["Wet", "Icy", "Snow covered"]:
        risk_score += 25
    
    # Alcohol factor
    if alcohol == "Yes":
        risk_score += 50
    
    # Fatigue factor
    if fatigue == "Yes":
        risk_score += 35
    
    # Speed factor (higher speeds increase risk)
    risk_score += min(30, (speed - 30) / 2) if speed > 30 else 0
    
    # Hour factor (late night hours increase risk)
    if hour >= 22 or hour <= 5:
        risk_score += 20
    
    # Normalize risk score to probability (0-100%)
    probability = min(95, max(5, risk_score)) / 100
    
    # Determine risk level
    if probability < 0.3:
        risk_level = "Low Risk"
        action = "Safe to proceed with normal caution."
    elif probability < 0.6:
        risk_level = "Moderate Risk"
        action = "Proceed with extra caution and reduced speed."
    else:
        risk_level = "High Risk"
        action = "Consider postponing travel or extreme caution required."
    
    return risk_level, f"{probability:.1%}", action

# Assuming X contains your feature columns
# In a real implementation, you would load your model and feature names
# For demo purposes, I'll create mock column names
X = pd.DataFrame(columns=[
    'Weather_Conditions_Clear', 'Weather_Conditions_Rain', 'Weather_Conditions_Snow', 'Weather_Conditions_Fog',
    'Road_Type_Urban', 'Road_Type_Highway', 'Road_Type_Rural',
    'Road_Conditions_Dry', 'Road_Conditions_Wet', 'Road_Conditions_Icy',
    'Alcohol_Involved_Yes', 'Alcohol_Involved_No',
    'Driver_Fatigue_Yes', 'Driver_Fatigue_No'
])

weather_opts = [col.replace('Weather_Conditions_', '') for col in X.columns if 'Weather_Conditions_' in col]
road_type_opts = [col.replace('Road_Type_', '') for col in X.columns if 'Road_Type_' in col]
road_cond_opts = [col.replace('Road_Conditions_', '') for col in X.columns if 'Road_Conditions_' in col]
alcohol_opts = [col.replace('Alcohol_Involved_', '') for col in X.columns if 'Alcohol_Involved_' in col]
fatigue_opts = [col.replace('Driver_Fatigue_', '') for col in X.columns if 'Driver_Fatigue_' in col]

with gr.Blocks() as demo:
    gr.Markdown("## 🚗 AI-Powered Traffic Risk Predictor")

    with gr.Row():
        weather = gr.Dropdown(weather_opts, label="Weather Conditions", value=weather_opts[0])
        road_type = gr.Dropdown(road_type_opts, label="Road Type", value=road_type_opts[0])
        road_cond = gr.Dropdown(road_cond_opts, label="Road Condition", value=road_cond_opts[0])

    with gr.Row():
        alcohol = gr.Dropdown(alcohol_opts, label="Alcohol Involved", value=alcohol_opts[1])  # Default to "No"
        fatigue = gr.Dropdown(fatigue_opts, label="Driver Fatigue", value=fatigue_opts[1])    # Default to "No"

    speed = gr.Number(label="Speed Limit (km/h)", value=40)
    hour = gr.Slider(0, 23, label="Hour of Day", value=12, step=1)

    with gr.Row():
        btn = gr.Button("Predict Risk")

    output1 = gr.Text(label="🧠 Risk Level")
    output2 = gr.Text(label="📊 Probability")
    output3 = gr.Text(label="✅ Suggested Action")

    btn.click(fn=predict_risk, inputs=[weather, road_type, road_cond, alcohol, fatigue, speed, hour],
              outputs=[output1, output2, output3])

demo.launch()
