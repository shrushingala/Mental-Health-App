from flask import Flask, render_template, request, Response
import pandas as pd
import joblib
import cv2
import os
app = Flask(__name__)

# Load models and encoders
mental_health_model = joblib.load('models/mental_health_model.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')  # Load all column encoders
label_encoder_target = joblib.load('models/mental_health_history_label_encoder.pkl')
emotion_model = joblib.load('models/emotion_recognition_model.pkl')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}
emotions_detected = []  # Initialize as an empty list
analyzing = False
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/submit-questionnaire', methods=['POST'])
def submit_questionnaire():
    # Collect form data from user responses
    data_dict = {
        'Age': request.form['age'], 
        'Gender': request.form['gender'], 
        'Occupation': request.form['occupation'],
        'Days_Indoors': request.form['days_indoors'],
        'Growing_Stress': request.form['growing_stress'],
        'Quarantine_Frustrations': request.form['quarantine_frustrations'],
        'Changes_Habits': request.form['changes_habits'],
        'Weight_Change': request.form['weight_change'],
        'Mood_Swings': request.form['mood_swings'],
        'Coping_Struggles': request.form['coping_struggles'],
        'Work_Interest': request.form['work_interest'],
        'Social_Weakness': request.form['social_weakness']
    }

    # Convert form data into a DataFrame
    df = pd.DataFrame([data_dict])

    # Encode string values into numerical values using saved LabelEncoders
    for column in df.columns:
        if column in label_encoders:
            le = label_encoders[column]  # Get the LabelEncoder for this column
            df[column] = le.transform(df[column])  # Transform the string values into numbers

    # Predict mental health history using the trained model
    numerical_prediction = mental_health_model.predict(df)[0]

    # Decode numerical prediction back to original category
    decoded_prediction = label_encoder_target.inverse_transform([numerical_prediction])[0]

    # Generate feedback based on prediction result
    feedback_mapping = {
        "Yes": "You may have a history of mental health challenges. Consider consulting a professional counselor.",
        "No": "You seem to have no significant history of mental health issues. Maintain a healthy lifestyle.",
        "Maybe": "There may be signs of mental health concerns. Monitor your well-being and seek help if needed, It’s okay to not feel okay—acknowledge your emotions and give yourself grace, Seeking help is a sign of strength, not weakness. You are not alone in this journey"
    }
    feedback = feedback_mapping.get(decoded_prediction)

    return render_template(
        "results.html",
        prediction=decoded_prediction,
        feedback=feedback,
    )

@app.route('/emotion-detection')
def emotion_detection():
    return render_template('emotion_detection.html')

def gen_frames():
    """Generate frames from the webcam for real-time emotion detection."""
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (48, 48)).flatten()
                prediction = emotion_model.predict([resized_face])[0]
                emotion_label = emotion_labels.get(prediction, "Unknown")  # Map prediction to label

                # Draw a rectangle around the face and display the predicted emotion
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    global analyzing
    analyzing = True    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop the emotion analysis."""
    global analyzing
    analyzing = False

    # Ensure 'static' directory exists
    if not os.path.exists("static"):
        os.makedirs("static")

    # Generate pie chart from detected emotions
    if emotions_detected:
        plt.figure(figsize=(6, 6))
        plt.pie(
            [emotions_detected.count(e) for e in set(emotions_detected)],
            labels=set(emotions_detected),
            autopct='%1.1f%%',
            startangle=140,
        )
        plt.title("Emotion Analysis")
        chart_path = os.path.join("static", "emotion_pie_chart.png")
        plt.savefig(chart_path)
        plt.close()
        return jsonify({"status": "stopped", "chart_path": chart_path})
    else:
        return jsonify({"status": "stopped", "chart_path": None})

@app.route('/results')
def results():
    """Display results after stopping analysis."""
    chart_path = os.path.join("static", "emotion_pie_chart.png")
    if os.path.exists(chart_path):
        return render_template('results.html', chart_path=chart_path)
    else:
        return render_template('results.html', chart_path=None)

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/dummy')
def dummy():
    return render_template('dummy.html')

if __name__ == "__main__":
    app.run(debug=True)
