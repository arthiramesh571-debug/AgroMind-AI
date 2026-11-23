from flask import Flask, render_template, request, redirect, send_from_directory, session, jsonify, url_for
from flask_mysqldb import MySQL
from gtts import gTTS
import os
import uuid
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "AgroMind_AI_Secret_Key_2026"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

# MySQL Configuration
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "AgroMind@2026"
app.config["MYSQL_DB"] = "AgroMind_AI"

mysql = MySQL(app)
# AI MODEL
MODEL_PATH = "models/tomato_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "bacterial_spot",
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites_two_spotted_spider_mite",
    "target_spot",
    "tomato_mosaic_virus",
    "tomato_yellow_leaf_curl_virus"
]


# Chilli Disease Model
CHILLI_MODEL_PATH = "models/chilli_disease_model.keras"

chilli_model = tf.keras.models.load_model(CHILLI_MODEL_PATH)
PADDY_MODEL_PATH = "models/paddy_disease_model.keras"
paddy_model = tf.keras.models.load_model(PADDY_MODEL_PATH)
BRINJAL_MODEL_PATH = "brinjal_leaf_training/brinjal_leaf_model.keras"
brinjal_model = tf.keras.models.load_model(BRINJAL_MODEL_PATH)

paddy_class_names = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

chilli_class_names = [
    "bacterial_spot",
    "healthy"
]
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        cursor = mysql.connection.cursor()

        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, password)
        )

        mysql.connection.commit()
        cursor.close()

        return redirect("/login")

    return render_template("register.html")
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        cursor = mysql.connection.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email = %s AND password = %s",
            (email, password)
        )

        user = cursor.fetchone()
        cursor.close()

        if user:
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            return redirect("/dashboard")

        return "<h3>Invalid email or password</h3><a href='/login'>Try Again</a>"

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():

    user_id = session.get("user_id")

    if not user_id:
        return redirect("/login")

    cursor = mysql.connection.cursor()

    cursor.execute(
        """
        SELECT c.crop_name, dh.health_score
        FROM detection_history dh
        JOIN crops c ON dh.crop_id = c.id
        WHERE dh.user_id = %s
        ORDER BY dh.detected_at DESC
        LIMIT 1
        """,
        (user_id,)
    )

    latest_detection = cursor.fetchone()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM detection_history
        WHERE user_id = %s
        """,
        (user_id,)
    )

    detection_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM alerts
        WHERE user_id = %s AND is_read = FALSE
        """,
        (user_id,)
    )

    alert_count = cursor.fetchone()[0]

    cursor.close()

    if latest_detection:
        selected_crop = latest_detection[0]
        crop_health = latest_detection[1]
    else:
        selected_crop = "No Detection"
        crop_health = 0

    return render_template(
        "dashboard.html",
        selected_crop=selected_crop,
        crop_health=crop_health,
        detection_count=detection_count,
        alert_count=alert_count
    )

@app.route("/detection", methods=["GET", "POST"])
def detection():

    if request.method == "POST":

        crop = request.form["crop"]
        image = request.files["leaf_image"]

        if image and image.filename:

            filename = secure_filename(image.filename)

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            image.save(image_path)

            # Select model based on crop
            if crop == "Tomato":
                selected_model = model
                selected_class_names = class_names

            elif crop == "Chilli":
                selected_model = chilli_model
                selected_class_names = chilli_class_names

            elif crop == "Paddy":
                selected_model = paddy_model
                selected_class_names = paddy_class_names
            
            elif crop == "Brinjal":
                selected_model = brinjal_model
                selected_class_names = [
                "Healthy_Leaf",
                "Insect_Pest_Disease",
                "Leaf_Spot_Disease",
                "Mosaic_Virus_Disease",
                "White_Mold_Disease",
                "Wilt_Disease"
            ]

            else:
                return "This crop is not supported yet."

            # Load and preprocess image
            if crop == "Brinjal":
              image_size = (160, 160)
            else:
              image_size = (224, 224)

            image_data = tf.keras.utils.load_img(
              image_path,
              target_size=image_size
            )

            image_array = tf.keras.utils.img_to_array(
                image_data
            )

           
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
            predictions = selected_model.predict(
                image_array,
                verbose=0
            )

            predicted_index = int(np.argmax(predictions[0]))

            disease_name = selected_class_names[predicted_index]

            confidence = float(predictions[0][predicted_index] * 100)

            print("Crop:", crop)
            print("Predicted class:", disease_name)
            print("AI confidence:", round(confidence, 2), "%")

            # Health score and severity
        
        if disease_name in [
            "healthy",
            "normal",
            "Healthy_Leaf"
        ]:
            health_score = 95
            severity = "Healthy"

        elif disease_name in [
            "bacterial_spot",
            "bacterial_leaf_blight",
            "bacterial_leaf_streak",
            "bacterial_panicle_blight",
            "brown_spot",
            "downy_mildew",
            "hispa"
        ]:
            health_score = 60
            severity = "Moderate"

        elif disease_name in [
            "Insect_Pest_Disease",
            "Leaf_Spot_Disease",
            "Mosaic_Virus_Disease",
            "White_Mold_Disease",
            "Wilt_Disease"
        ]:
            health_score = 60
            severity = "Moderate"

        elif disease_name in [
            "early_blight",
            "late_blight",
            "blast",
            "dead_heart",
            "tungro"
        ]:
            health_score = 50
            severity = "Severe"

        elif disease_name == "leaf_mold":
            health_score = 65
            severity = "Moderate"

        else:
            health_score = 70
            severity = "Mild"

        # Disease information
        disease_info = {
                "early_blight": "Early blight is a common fungal disease that affects tomato leaves and can reduce plant growth and yield.",
                "late_blight": "Late blight is a serious disease that can rapidly damage tomato leaves, stems and fruits.",
                "bacterial_spot": "Bacterial spot causes dark spots on leaves and may affect fruits.",
                "leaf_mold": "Leaf mold mainly affects tomato leaves under warm and humid conditions.",
                "healthy": "The leaf appears healthy with no major disease symptoms detected.",

                "Insect_Pest_Disease": "The leaf shows signs associated with insect pest damage. Inspect the plant carefully.",
                "Leaf_Spot_Disease": "Leaf spot symptoms may appear as spots on the brinjal leaf.",
                "Mosaic_Virus_Disease": "Mosaic-like patterns or uneven leaf coloration may be present.",
                "White_Mold_Disease": "White mold-like growth may be associated with this classification.",
                "Wilt_Disease": "Wilting symptoms may indicate a plant health problem.",

                # Paddy
                "bacterial_leaf_blight": "Bacterial leaf blight is a bacterial disease that causes water-soaked lesions and drying of rice leaves.",
                "bacterial_leaf_streak": "Bacterial leaf streak causes narrow, water-soaked streaks along the rice leaf surface.",
                "bacterial_panicle_blight": "Bacterial panicle blight can affect rice panicles and reduce grain development.",
                "blast": "Rice blast is a fungal disease that can affect leaves, nodes and panicles and may reduce crop yield.",
                "brown_spot": "Brown spot causes brown lesions on rice leaves and can affect plant growth and grain quality.",
                "dead_heart": "Dead heart is a condition where the central shoot of a young rice plant dries and dies.",
                "downy_mildew": "Downy mildew can cause pale or yellowish areas on rice leaves, particularly under favorable humid conditions.",
                "hispa": "Rice hispa is an insect pest that damages the leaf surface and can reduce photosynthetic activity.",
                "normal": "The rice leaf appears normal with no major disease symptoms detected.",
                "tungro": "Rice tungro is a viral disease that can cause yellowing, stunting and reduced growth."
            }

        symptoms = {
                "early_blight": "Brown circular spots, yellowing leaves and dark lesions may appear on older leaves.",
                "late_blight": "Dark water-soaked patches and rapidly spreading brown lesions may appear.",
                "bacterial_spot": "Small dark spots and yellowing around infected areas may be visible.",
                "leaf_mold": "Yellow patches on the upper leaf surface with mold growth underneath may occur.",
                "healthy": "No significant disease symptoms were detected.",

                "Insect_Pest_Disease": "Possible leaf damage or feeding marks may be visible.",
                "Leaf_Spot_Disease": "Spots or lesions may appear on the leaf surface.",
                "Mosaic_Virus_Disease": "Mosaic-like patterns or uneven coloration may appear on leaves.",
                "White_Mold_Disease": "White mold-like growth may be visible on affected plant parts.",
                "Wilt_Disease": "Leaves may droop, lose firmness or wilt.",


                # Paddy
                "bacterial_leaf_blight": "Water-soaked lesions may appear along leaf margins, followed by yellowing and drying of leaves.",
                "bacterial_leaf_streak": "Long narrow water-soaked streaks may appear between the leaf veins.",
                "bacterial_panicle_blight": "Panicles may show discoloration, poor grain development and affected spikelets.",
                "blast": "Spindle-shaped brown or gray lesions may appear on leaves, while neck infection can affect panicles.",
                "brown_spot": "Small circular or oval brown spots may develop on rice leaves.",
                "dead_heart": "The central shoot dries and can be pulled out easily from the plant.",
                "downy_mildew": "Yellowish or pale areas may appear on leaves, sometimes with downy growth under humid conditions.",
                "hispa": "Scraped leaf surfaces and white streaks may appear due to feeding damage.",
                "normal": "The leaf appears green and healthy without significant visible symptoms.",
                "tungro": "Leaves may become yellow or orange-yellow and plants may show stunted growth."
            }

        treatment = {
                "early_blight": "Remove infected leaves, improve air circulation and avoid watering the leaves directly.",
                "late_blight": "Remove infected plant parts and maintain good air circulation.",
                "bacterial_spot": "Remove infected leaves and avoid overhead watering.",
                "leaf_mold": "Improve ventilation, reduce humidity and remove heavily infected leaves.",
                "healthy": "Continue regular monitoring, proper watering and good crop management practices.",

                "Insect_Pest_Disease": "Inspect the plant for pests and follow locally recommended integrated pest management practices.",
                "Leaf_Spot_Disease": "Remove severely affected leaves where practical, avoid unnecessary leaf wetness and maintain good plant spacing.",
                "Mosaic_Virus_Disease": "Remove severely affected plants where appropriate and manage insect vectors using recommended practices.",
                "White_Mold_Disease": "Remove affected plant material where practical and reduce excessive moisture around plants.",
                "Wilt_Disease": "Check soil moisture and root health; remove severely affected plants where appropriate.",


                # Paddy
                "bacterial_leaf_blight": "Remove severely affected plant material, maintain proper field sanitation and manage irrigation carefully.",
                "bacterial_leaf_streak": "Maintain field sanitation, avoid unnecessary leaf wetness and use healthy planting material.",
                "bacterial_panicle_blight": "Remove affected plant material where practical and maintain proper field sanitation and crop management.",
                "blast": "Use healthy planting material, maintain balanced nitrogen application and follow recommended disease management practices.",
                "brown_spot": "Maintain balanced crop nutrition, improve field management and remove heavily affected plant material where practical.",
                "dead_heart": "Remove affected shoots where practical and manage the crop according to recommended pest-control practices.",
                "downy_mildew": "Improve field ventilation, avoid excessive moisture and remove severely affected plant material where practical.",
                "hispa": "Remove heavily affected leaves where practical and follow recommended integrated pest-management practices.",
                "normal": "Continue regular monitoring, proper irrigation and balanced crop management.",
                "tungro": "Remove severely affected plants where practical and manage the vector according to recommended integrated pest-management practices."
            }

        prevention = {
                "early_blight": "Use healthy planting material, maintain proper spacing and remove infected plant debris.",
                "late_blight": "Avoid excessive moisture, maintain good spacing and regularly inspect leaves.",
                "bacterial_spot": "Use disease-free seeds, avoid wet foliage and maintain field sanitation.",
                "leaf_mold": "Maintain proper ventilation, reduce humidity and avoid excessive irrigation.",
                "healthy": "Continue regular monitoring and maintain balanced irrigation and nutrition.",

                "Insect_Pest_Disease": "Monitor plants regularly, inspect leaf undersides and use integrated pest management practices.",
                "Leaf_Spot_Disease": "Use healthy planting material, maintain good spacing and avoid prolonged leaf wetness.",
                "Mosaic_Virus_Disease": "Use healthy seedlings, remove suspicious plants as appropriate and monitor insect vectors.",
                "White_Mold_Disease": "Maintain suitable spacing, improve airflow and avoid excessive soil moisture.",
                "Wilt_Disease": "Use healthy planting material, maintain good soil drainage and monitor plants for early symptoms.",

                # Paddy
                "bacterial_leaf_blight": "Use healthy planting material, maintain field sanitation and avoid excessive nitrogen application.",
                "bacterial_leaf_streak": "Use healthy planting material, maintain field sanitation and monitor the crop regularly.",
                "bacterial_panicle_blight": "Use healthy planting material, maintain good field sanitation and monitor panicles during crop development.",
                "blast": "Use healthy planting material, maintain balanced nutrition and regularly monitor leaves and panicles.",
                "brown_spot": "Maintain balanced nutrition, proper water management and good field sanitation.",
                "dead_heart": "Monitor the crop regularly, maintain field sanitation and use integrated pest-management practices.",
                "downy_mildew": "Avoid excessive moisture, maintain good crop spacing and regularly inspect leaves.",
                "hispa": "Monitor leaves regularly, maintain field sanitation and use integrated pest-management practices.",
                "normal": "Continue regular monitoring, balanced irrigation and proper crop nutrition.",
                "tungro": "Use healthy planting material, monitor the crop regularly and manage disease vectors using integrated pest-management practices."
            }

           
        # Save detection result to database
        user_id = session.get("user_id")

        cursor = mysql.connection.cursor()

        cursor.execute(
                "SELECT id FROM crops WHERE crop_name = %s",
                (crop,)
            )

        crop_data = cursor.fetchone()

        if user_id and crop_data:
                crop_id = crop_data[0]

                cursor.execute(
                    """
                    INSERT INTO detection_history
                    (user_id, crop_id, image_path, disease_name,
                     confidence, severity, health_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        crop_id,
                        image_path,
                        disease_name,
                        confidence,
                        severity,
                        health_score
                    )
                )

                # Automatically create Smart Alerts
                if health_score < 50:
                    cursor.execute(
                        """
                        INSERT INTO alerts
                        (user_id, alert_type, message, severity)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            "Critical Crop Health",
                            f"{crop} health score is {health_score}/100. Immediate attention is recommended.",
                            "Critical"
                        )
                    )

                elif health_score < 70:
                    cursor.execute(
                        """
                        INSERT INTO alerts
                        (user_id, alert_type, message, severity)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            "Crop Health Warning",
                            f"{crop} health score is {health_score}/100. Please monitor the crop closely.",
                            "Warning"
                        )
                    )

                if disease_name not in ["healthy", "normal"]:
                    cursor.execute(
                        """
                        INSERT INTO alerts
                        (user_id, alert_type, message, severity)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            "Disease Detected",
                            f"{disease_name} was detected in your {crop} crop.",
                            "Warning"
                        )
                    )

                mysql.connection.commit()

        cursor.close()

        return render_template(
                "result.html",
                crop=crop,
                disease_name=disease_name,
                confidence=confidence,
                health_score=health_score,
                severity=severity,
                disease_info=disease_info.get(
                    disease_name,
                    "Disease information is currently unavailable."
                ),
                symptoms=symptoms.get(
                    disease_name,
                    "Symptoms information is currently unavailable."
                ),
                treatment=treatment.get(
                    disease_name,
                    "Treatment information is currently unavailable."
                ),
                prevention=prevention.get(
                    disease_name,
                    "Prevention information is currently unavailable."
                ),
                image_path=image_path
            )

    return render_template("disease_detection.html")


@app.route("/history")
def history():
    user_id = session.get("user_id")

    if not user_id:
        return redirect("/login")

    cursor = mysql.connection.cursor()

    cursor.execute(
        """
        SELECT
            dh.id,
            c.crop_name,
            dh.image_path,
            dh.disease_name,
            dh.confidence,
            dh.severity,
            dh.health_score,
            dh.detected_at
        FROM detection_history dh
        JOIN crops c
        ON dh.crop_id = c.id
        WHERE dh.user_id = %s
        ORDER BY dh.detected_at DESC
        """,
        (user_id,)
    )

    history_data = cursor.fetchall()

    cursor.close()

    return render_template(
        "history.html",
        history=history_data
    )

@app.route("/crop-knowledge")
def crop_knowledge():
    return render_template("crop_knowledge.html")

@app.route("/crop-health")
def crop_health():

    user_id = session.get("user_id")

    if not user_id:
        return redirect("/login")

    cursor = mysql.connection.cursor()

    cursor.execute(
        """
        SELECT
            c.crop_name,
            dh.disease_name,
            dh.confidence,
            dh.severity,
            dh.health_score,
            dh.detected_at
        FROM detection_history dh
        JOIN crops c
        ON dh.crop_id = c.id
        WHERE dh.user_id = %s
        ORDER BY dh.detected_at DESC
        LIMIT 1
        """,
        (user_id,)
    )

    latest = cursor.fetchone()

    cursor.close()

    if latest:

        crop_name = latest[0]
        disease_name = latest[1]
        confidence = round(float(latest[2]), 2)
        severity = latest[3]
        health_score = float(latest[4])
        detected_at = latest[5]

        if health_score >= 80:
            health_status = "Excellent Crop Health 🌱"

        elif health_score >= 60:
            health_status = "Moderate Crop Health 🟡"

        else:
            health_status = "Attention Required 🔴"

    else:

        crop_name = "No Detection"
        disease_name = "No Disease Detected"
        confidence = 0
        severity = "Not Available"
        health_score = 0
        detected_at = "No previous detection"

        health_status = "No Health Data Available"

    return render_template(
        "crop_health.html",
        crop_name=crop_name,
        disease_name=disease_name,
        confidence=confidence,
        severity=severity,
        health_score=health_score,
        detected_at=detected_at,
        health_status=health_status
    )

@app.route("/alerts")
def alerts():

    user_id = session.get("user_id")

    if not user_id:
        return redirect("/login")

    cursor = mysql.connection.cursor()

    # Get all alerts for the logged-in user
    cursor.execute(
        """
        SELECT
            id,
            alert_type,
            message,
            severity,
            is_read,
            created_at
        FROM alerts
        WHERE user_id = %s
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    alerts_data = cursor.fetchall()

    # Count critical alerts
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM alerts
        WHERE user_id = %s
        AND LOWER(severity) = 'critical'
        """,
        (user_id,)
    )

    critical_count = cursor.fetchone()[0]

    # Count warning alerts
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM alerts
        WHERE user_id = %s
        AND LOWER(severity) = 'warning'
        """,
        (user_id,)
    )

    warning_count = cursor.fetchone()[0]

    # Count information alerts
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM alerts
        WHERE user_id = %s
        AND LOWER(severity) = 'info'
        """,
        (user_id,)
    )

    info_count = cursor.fetchone()[0]

    cursor.close()

    total_count = len(alerts_data)

    return render_template(
        "alerts.html",
        alerts=alerts_data,
        critical_count=critical_count,
        warning_count=warning_count,
        info_count=info_count,
        total_count=total_count
    )
@app.route("/voice-assistant")
def voice_assistant():
    return render_template("voice_assistant.html")

@app.route("/speak-tamil", methods=["POST"])
def speak_tamil():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is empty"}), 400

    audio_folder = os.path.join(app.static_folder, "audio")
    os.makedirs(audio_folder, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(audio_folder, filename)

    try:
        gTTS(text=text, lang="ta").save(filepath)
        return jsonify({
            "audio_url": url_for("static", filename=f"audio/{filename}")
        })
    except Exception as e:
        return jsonify({"error": "Tamil audio generation failed. Check internet connection."}), 500
    


import random
from datetime import datetime

@app.route("/iot-monitor")
def iot_monitor():
    temperature = round(random.uniform(24, 38), 1)
    humidity = round(random.uniform(40, 90), 1)
    soil_moisture = round(random.uniform(20, 80), 1)

    if soil_moisture < 30:
        alert = "Soil moisture is low. Check irrigation needs."
    else:
        alert = "Soil moisture is within the demo range."

    return render_template(
        "iot_monitor.html",
        temperature=temperature,
        humidity=humidity,
        soil_moisture=soil_moisture,
        alert=alert,
        updated_at=datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
    )

if __name__ == "__main__":
    app.run(debug=True)