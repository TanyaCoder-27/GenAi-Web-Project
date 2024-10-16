'''
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_module import predict_disease

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms')
    
    if symptoms:
        try:
            disease, precautions, description, foods_to_take, foods_to_avoid = predict_disease(symptoms)
            return jsonify({
                'disease': disease,
                'precautions': precautions,
                'description': description,
                'foods_to_take': foods_to_take,
                'foods_to_avoid': foods_to_avoid
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'No symptoms provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
'''

'''
#This worked for the first html code of selection symptoms set from dataset
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and other required files
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
disease_info = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Info.csv", encoding='ISO-8859-1')

# Load symptoms data
symptoms_data = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Symptoms.csv", encoding='ISO-8859-1')
all_symptoms = symptoms_data['Symptoms'].tolist()

@app.route('/')
def home():
    return render_template('index.html', symptoms=all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    # Get symptoms from the form (from HTML)
    input_symptoms = request.form.getlist('symptoms')
    
    # Convert symptoms to a format the model can understand
    input_data = vectorizer.transform([' '.join(input_symptoms)]).toarray()
    
    # Predict the disease
    predicted_disease_encoded = model.predict(input_data)
    predicted_disease = label_encoder.inverse_transform(predicted_disease_encoded)[0]
    
    # Get the detailed information of the predicted disease
    disease_details = disease_info[disease_info['Disease'] == predicted_disease].iloc[0]
    
    # Prepare data to send back to frontend
    response = {
        'disease': predicted_disease,
        'precautions': disease_details['Precautions'],
        'description': disease_details['Description'],
        'foods_to_take': disease_details['Foods to take'],
        'foods_to_avoid': disease_details['Foods to avoid']
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
'''



'''
#this worked for the Yajnica's frontent but backend not worked properly
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and other required files
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
disease_info = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Info.csv", encoding='ISO-8859-1')

# Load symptoms data
symptoms_data = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Symptoms.csv", encoding='ISO-8859-1')
all_symptoms = symptoms_data['Symptoms'].tolist()

@app.route('/')
def home():
    return render_template('index.html', symptoms=all_symptoms)

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input (symptom or message)
    user_input = request.json['message']
    
    # Check if the input is a symptom (basic check, you can enhance this logic)
    if user_input in all_symptoms:
        # Convert symptom to a format the model can understand
        input_data = vectorizer.transform([user_input]).toarray()
        
        # Predict the disease
        predicted_disease_encoded = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(predicted_disease_encoded)[0]
        
        # Get the detailed information of the predicted disease
        disease_details = disease_info[disease_info['Disease'] == predicted_disease].iloc[0]
        
        # Prepare the response message
        response = f"The predicted disease is: {predicted_disease}. " \
                   f"Precautions: {disease_details['Precautions']}. " \
                   f"Description: {disease_details['Description']}. " \
                   f"Foods to take: {disease_details['Foods to take']}. " \
                   f"Foods to avoid: {disease_details['Foods to avoid']}."
    else:
        response = "I'm sorry, I couldn't understand your input. Please select a symptom or provide a clear message."

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
'''

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from chatbot_module import predict_disease  # Importing the function directly

app = Flask(__name__)

# Load trained model and other required files
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
disease_info = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Info.csv", encoding='ISO-8859-1')

# Load symptoms data
symptoms_data = pd.read_csv("C:/Users/tanya/OneDrive/Desktop/Latest of Latest MCCP DAY MINI PROJECT/New_Disease_Symptoms.csv", encoding='ISO-8859-1')
all_symptoms = symptoms_data['Symptoms'].tolist()

@app.route('/')
def home():
    return render_template('index.html', symptoms=all_symptoms)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    
    if user_input:
        try:
            predicted_disease, precautions, description, foods_to_take, foods_to_avoid = predict_disease(user_input)
            
            # Prepare the response message
            response = f"The predicted disease is: {predicted_disease}. " \
                       f"Precautions: {', '.join(precautions)}. " \
                       f"Description: {description}. " \
                       f"Foods to take: {', '.join(foods_to_take)}. " \
                       f"Foods to avoid: {', '.join(foods_to_avoid)}."
        except Exception as e:
            response = "Sorry, an error occurred while processing your request."
            print(e)  # Log the exception for debugging
    else:
        response = "I'm sorry, I couldn't understand your input. Please select a symptom or provide a clear message."

    return jsonify({'response': response})

# New Route for File Upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"response": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"response": "No selected file"}), 400

    # Save and process the file (logic will be added later)
    file.save(f"uploads/{file.filename}")
    return jsonify({"response": "File successfully uploaded!"})

if __name__ == "__main__":
    app.run(debug=True)
