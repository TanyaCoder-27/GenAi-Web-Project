import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load disease data and models
disease_info_data = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Info.csv", encoding='ISO-8859-1')
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Predict disease
def predict_disease(symptoms):
    symptoms = preprocess_text(symptoms)
    symptoms_vectorized = vectorizer.transform([symptoms])
    prediction = svm_model.predict(symptoms_vectorized)
    predicted_disease = label_encoder.inverse_transform(prediction)[0]

    # Fetch disease details
    disease_row = disease_info_data[disease_info_data['Disease'] == predicted_disease]
    precautions = disease_row['Precautions'].values[0].split(',')
    description = disease_row['Description'].values[0]
    foods_to_take = disease_row['Foods to take'].values[0].split(',')
    foods_to_avoid = disease_row['Foods to avoid'].values[0].split(',')

    return predicted_disease, precautions, description, foods_to_take, foods_to_avoid
