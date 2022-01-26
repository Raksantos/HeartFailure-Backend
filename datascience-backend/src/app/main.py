from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS
import os

columns=["Idade (anos)", "Pressão sanguínea em repouso (mm Hg)",
       "Colesterol (mm/dl)", "Açúcar no sangue em jejum",
       "Frequência cardíaca máxima alcançada", "Oldpeak", "Sexo_F", "Sexo_M",
       "Tipo de dor no peito_ASY", "Tipo de dor no peito_ATA",
       "Tipo de dor no peito_NAP", "Tipo de dor no peito_TA",
       "Resultados de eletrocardiograma em repouso_LVH",
       "Resultados de eletrocardiograma em repouso_Normal",
       "Resultados de eletrocardiograma em repouso_ST",
       "Angina induzida por exercício_N", "Angina induzida por exercício_Y",
       "Inclinação do segmento ST de pico do exercício_Down",
       "Inclinação do segmento ST de pico do exercício_Flat",
       "Inclinação do segmento ST de pico do exercício_Up"]

model = pickle.load(open('../../models/knn_model.sav', 'rb'))
ro_scaler = pickle.load(open('../../models/ro_scaler.sav', 'rb'))

app = app = Flask(__name__)
cors = CORS(app, resource={r"/*":{"origins": "*"}})

@app.route('/')
def home():
    return '<h1>Data Science Project API</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    print(data)

    data_input = [data[col] for col in columns]

    df = pd.DataFrame(data=[data_input], columns=columns)

    X_valid = ro_scaler.transform(df)

    result = model.predict(X_valid)

    result = result.tolist()

    return jsonify(Result=result[0])

if __name__ == "__main__":
  port = os.getenv("PORT", 5000)
  app.run(debug=False, host="0.0.0.0", port=port)
