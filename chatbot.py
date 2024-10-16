import pandas as pd
import nltk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from sklearn.metrics import accuracy_score  # Import for accuracy calculation

# Load datasets
disease_symptoms_data = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Symptoms.csv", encoding='ISO-8859-1')
disease_info_data = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Info.csv", encoding='ISO-8859-1')

# Preprocessing text
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

disease_symptoms_data['Symptoms'] = disease_symptoms_data['Symptoms'].apply(preprocess_text)

# Vectorizing symptoms text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(disease_symptoms_data['Symptoms'])
y = disease_symptoms_data['Disease']

# Encoding disease labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Save the model, vectorizer, and encoder
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# **Calculate and print the model's accuracy**
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Function to predict disease and provide additional information
def predict_disease(symptoms):
    # Load the saved model and tools
    svm_model = joblib.load('svm_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Preprocess and predict
    symptoms = preprocess_text(symptoms)
    symptoms_vectorized = vectorizer.transform([symptoms])
    prediction = svm_model.predict(symptoms_vectorized)
    predicted_disease = label_encoder.inverse_transform(prediction)[0]

    # Fetch disease information from the second dataset
    disease_row = disease_info_data.loc[disease_info_data['Disease'] == predicted_disease]
    precautions = disease_row['Precautions'].values[0].split(',')
    description = disease_row['Description'].values[0]
    foods_to_take = disease_row['Foods to take'].values[0].split(',')
    foods_to_avoid = disease_row['Foods to avoid'].values[0].split(',')

    return predicted_disease, precautions, description, foods_to_take, foods_to_avoid
