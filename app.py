from flask import Flask, render_template, request
import pickle
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Load model terlatih
model = pickle.load(open('model(3).pkl', 'rb'))

# Halaman utama
@app.route('/')
def home():
    return render_template('home.html')

# Endpoint untuk memproses prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil nilai input dari form
        amount = float(request.form['amount'])
        transaction_type = int(request.form['transaction_type'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Susun input untuk model
        input_data = np.array([[amount, transaction_type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])

        # Prediksi
        prediction = model.predict(input_data)[0]

        # Interpretasi hasil
        result = "Penipuan" if prediction == 1 else "Bukan Penipuan"

        return render_template('result.html', prediction=result, form_data=request.form)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", form_data=None)

# Jalankan app
if __name__ == '__main__':
    app.run(debug=True)
