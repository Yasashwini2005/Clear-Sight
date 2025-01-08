from flask import Flask, request, render_template
import mysql.connector
from PIL import Image
import io
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("glaucoma_model.h5")


CLASS_LABELS = {0: "No Glaucoma", 1: "Glaucoma Detected"}

# Flask app
app = Flask(__name__)

# MySQL configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'yash',
    'database': 'glaucoma_db'
}

# Connect to the database
try:
    db_connection = mysql.connector.connect(**db_config)
    cursor = db_connection.cursor()
except mysql.connector.Error as e:
    raise RuntimeError(f"Database connection failed: {e}")

# Email configuration
SMTP_SERVER = "smtp.gmail.com"  # Replace with your SMTP server
SMTP_PORT = 587
EMAIL_ADDRESS = "yasashwini31@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "xxxx"  # Replace with your email password

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        description = request.form.get('description', '').strip()

        if 'image' not in request.files:
            return render_template("index.html",
                                   error="No file uploaded. Please upload an image.",
                                   name=name, email=email, age=age, gender=gender, description=description)

        file = request.files['image']
        if file.filename == '':
            return render_template("index.html",
                                   error="No file selected. Please choose an image file.",
                                   name=name, email=email, age=age, gender=gender, description=description)

        # Preprocess the uploaded image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        processed_image = preprocess_image(image)

        # Make prediction using the loaded model
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction))  # Assuming a softmax output
        class_label = CLASS_LABELS.get(predicted_class, "Unknown")

        # Insert details into the database
        sql_query = """
        INSERT INTO user_details (name, email, age, gender, description, prediction)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql_query, (name, email, age, gender, description, class_label))
        db_connection.commit()

        # Send result via email
        subject = "CLEAR SIGHT:Glaucoma Prediction Result"
        message = f"""
        Hello {name},

        Thank you for using CLEAR SIGHT. Here are your results:

        Name: {name}
        Age: {age}
        Gender: {gender}
        Description: {description}
        Prediction: {class_label}

        Regards,
        CLEAR SIGHT Team
        """

        try:
            # Create the email
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))

            # Send the email
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
        except Exception as email_error:
            return render_template("index.html", error=f"Failed to send email: {email_error}")

        # Prepare the result
        result = {
            'name': name,
            'email': email,
            'age': age,
            'gender': gender,
            'description': description,
            'prediction': class_label
        }

        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("index.html",
                               error=f"An error occurred: {str(e)}",
                               name=name, email=email, age=age, gender=gender, description=description)

def preprocess_image(image):
    """
    Preprocess the input image for the model.
    """
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

if __name__ == "__main__":
    app.run(debug=True)
