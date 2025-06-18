import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS

# Spark
# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.classification import LogisticRegressionModel

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'  # Dossier pour stocker les fichiers uploadés
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Extensions autorisées
IMG_SIZE = (150, 150)  # Taille cible des images

# Crée le dossier d'upload s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialisation de la session Spark
# spark = SparkSession.builder \
#     .appName("PneumoniaDetection") \
#     .config("spark.executor.memory", "2g") \
#     .config("spark.driver.memory", "2g") \
#     .getOrCreate()

# Charge le modèle Keras pré-entraîné
keras_model = load_model('pneumonia_detection_model.h5')

# Spark ML model
# spark_model = LogisticRegressionModel.load("path/to/spark/model")

def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    """Prédit la pneumonie à partir d'une image en utilisant Keras"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = keras_model.predict(img_array)[0][0]
    return {
        'prediction': float(prediction),
        'diagnosis': 'Pneumonia' if prediction > 0.5 else 'Normal',
        'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
    }

# Fonction Spark
# def process_with_spark(data):
#     df = spark.createDataFrame([data])
#     return df.collect()

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for pneumonia prediction"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # 1. Prédiction avec Keras
            keras_result = predict_image(filepath)

            # Traitement Spark
            # spark_data = {
            #     'prediction': keras_result['prediction'],
            #     'timestamp': str(datetime.now()),
            # }
            # spark_result = process_with_spark(spark_data)

            # Résultat final sans Spark
            final_result = {
                'keras_prediction': keras_result,
                # 'spark_processing': str(spark_result),
                'combined_diagnosis': keras_result['diagnosis']
            }

            response = jsonify(final_result)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        # spark.stop()  # Spark stop
        pass
