from flask import Flask, render_template, request, jsonify
import pyodbc
import pickle


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.models import load_model

import numpy as np

app = Flask(__name__)

def get_db_connection(database_name):
    s = 'APB-JBS02-02L'
    u = 'GENERALADMINISTRATOR'
    p = 'GENERALADMIN@12345'
    cstr = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={s};DATABASE={database_name};UID={u};PWD={p}'
    conn = pyodbc.connect(cstr)
    return conn

# Load the deep learning model
model = load_model('D:/SHARING PROJECTS/CHIETA/CHATBOT/Model/DEEP LEARNING/model_without_batch_shape.keras')

# Load the TfidfVectorizer and LabelEncoder
with open('D:/SHARING PROJECTS/CHIETA/CHATBOT/Model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('D:/SHARING PROJECTS/CHIETA/CHATBOT/Model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def contains_gm_keyword(user_input):
    conn = get_db_connection('ChatbotPredictionDB')
    cursor = conn.cursor()
    cursor.execute("SELECT GrantKeyword FROM dbo.GMKeywords")
    keywords = [row[0].lower() for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return any(keyword in user_input.lower() for keyword in keywords)

def process_user_message(user_input):
    try:
        # Transform user input using the loaded TF-IDF vectorizer
        transformed_input = vectorizer.transform([user_input]).toarray()

        # Make a prediction with the deep learning model
        predictions = model.predict(transformed_input)

        # Get the class with the highest probability
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

        # Connect to ChatbotPredictionDB and check ResearchDepartment table
        conn = get_db_connection('ChatbotPredictionDB')
        cursor = conn.cursor()
        cursor.execute("SELECT ModelPrediction FROM ResearchDepartment WHERE ModelPrediction = ?", (predicted_class_label,))
        model_prediction_result = cursor.fetchone()

        if model_prediction_result:
            # If prediction exists in ModelPrediction, retrieve all PredictionProcess and PredictionURL entries
            cursor.execute("SELECT DISTINCT PredictionProcess, PredictionURL FROM ResearchDepartment WHERE ModelPrediction = ?", (predicted_class_label,))
            prediction_data = cursor.fetchall()
            prediction_processes = [row[0] for row in prediction_data]
            prediction_url = [row[1] for row in prediction_data]  # URL for the first prediction process
            cursor.close()
            conn.close()

            # HTML response structure for displaying PredictionProcess and URL
            response = f"<p>Do you mean '{predicted_class_label}'? <br> Here are the options I found for you:<br> {prediction_processes[0]}<br><br></p>"
            response += f"<a href='{prediction_url[0]}' target='_blank'>Click to read more</a><br>"
            response += "<br>".join(prediction_processes)
        else:
            # If prediction is not found in ResearchDepartment, use original functionality
            cursor.close()
            conn.close()
            
            conn2 = get_db_connection('ChatbotPredictionDB')
            cursor2 = conn2.cursor()
            query_url = "SELECT URL FROM dbo.PredictionURL WHERE TextContent = ?"
            cursor2.execute(query_url, (predicted_class_label,))
            url_result = cursor2.fetchone()
            cursor2.close()
            conn2.close()

            if url_result:
                url = url_result[0]
                conn3 = get_db_connection('grantManagementDB')
                
                cursor3 = conn3.cursor()
                cursor3.execute("SELECT DISTINCT funding_window_name FROM dbo.dgMaster WHERE programmes_afs = ?", (predicted_class_label,))
                funding_windows = [row[0] for row in cursor3.fetchall()]
                cursor3.close()
                conn3.close()

                if funding_windows:
                    response = "Here are the options I found. You can click to read more:<br>"
                    for window in funding_windows:
                        response += f"<a href='{url}' target='_blank'>{window}</a><br>"
                else:
                    response = "No matching funding windows were found for your query."
            else:
                response = "No URL found for the predicted option."
    except Exception as e:
        print(f"Error: {e}")
        response = "I encountered an error while processing your request. Please try again."

    return response

@app.route('/')
def index():
    return render_template('index.html')

def verify_user_info(email, username, department, company):
    conn = get_db_connection('ChatbotPredictionDB')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Email, Username, Department
        FROM dbo.GMVerificationLogin
        WHERE Email = ? AND Username = ? AND Department = ?
    """, (email, username, department))
    verification_result = cursor.fetchone()
    cursor.close()
    conn.close()

    if verification_result:
        conn = get_db_connection('grantManagementDB')
        cursor = conn.cursor()
        cursor.execute("SELECT applicationStatus FROM dbo.mgMaster WHERE organisationName = ?", (company,))
        app_status = cursor.fetchone()
        cursor.close()
        conn.close()

        if app_status:
            return f"Verification successful! Application status for {company}: {app_status[0]}"
        else:
            return f"Verification successful! But no application status found for {company}."
    else:
        return "Verification failed. Please check your input details."

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    if contains_gm_keyword(user_input):
        return jsonify({'response': "Please provide your email, username, department, and company name.", 'show_form': True})
    response = process_user_message(user_input)
    return jsonify({'response': response, 'show_form': False})

@app.route('/submit_additional_info', methods=['POST'])
def submit_additional_info():
    data = request.json
    email = data.get('email')
    username = data.get('username')
    department = data.get('department')
    company = data.get('company')
    
    response = verify_user_info(email, username, department, company)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
