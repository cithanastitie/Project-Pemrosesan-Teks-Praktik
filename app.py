import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# =========================
# LOAD MODEL & VECTORIZER
# =========================
def load_model_and_vectorizer():
    try:
        model = joblib.load('cyberbullying_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')

        print("✅ Model dan TF-IDF vectorizer berhasil dimuat!")
        return model, vectorizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


# Load saat aplikasi dijalankan
model, vectorizer = load_model_and_vectorizer()

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text', '')

        if not text.strip():
            return render_template(
                'index.html',
                error="Masukkan teks terlebih dahulu!"
            )

        # Transform teks ke TF-IDF
        text_vectorized = vectorizer.transform([text])

        # Prediksi
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]

        label = "Bullying" if prediction == 1 else "Tidak Bullying"
        confidence = round(float(prediction_proba[prediction]) * 100, 2)
        color = "danger" if prediction == 1 else "success"

        # Penjelasan & saran
        if prediction == 1:
            explanation = (
                "Teks mengandung indikasi bullying berdasarkan pola "
                "yang dipelajari oleh model machine learning."
            )
            suggestions = [
                "Hindari penggunaan kata-kata kasar atau menghina",
                "Gunakan bahasa yang lebih sopan dan konstruktif",
                "Pertimbangkan dampak kata-kata terhadap orang lain"
            ]
        else:
            explanation = (
                "Teks tidak mengandung indikasi bullying "
                "berdasarkan analisis model."
            )
            suggestions = [
                "Pertahankan komunikasi yang sehat",
                "Tetap perhatikan pilihan kata",
                "Hindari kata-kata yang berpotensi disalahartikan"
            ]

        return render_template(
            'result.html',
            text=text,
            prediction=label,
            confidence=confidence,
            color=color,
            explanation=explanation,
            suggestions=suggestions
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Terjadi error: {str(e)}"
        )


# =========================
# API ENDPOINT (OPSIONAL)
# =========================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Text is required'}), 400

        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        prediction_proba = model.predict_proba(text_vectorized)[0]

        return jsonify({
            'text': text,
            'prediction': 'Bullying' if prediction == 1 else 'Not Bullying',
            'confidence': float(prediction_proba[prediction]),
            'probabilities': {
                'not_bullying': float(prediction_proba[0]),
                'bullying': float(prediction_proba[1])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =========================
# HEALTH CHECK
# =========================
@app.route('/health')
def health_check():
    if model is not None and vectorizer is not None:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'vectorizer_loaded': True
        })
    return jsonify({'status': 'unhealthy'}), 500


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
