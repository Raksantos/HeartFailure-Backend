from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle


data = [[40,
 140,
 289,
 0,
 172,
 0,
 0,
 1,
 0,
 1,
 0,
 0,
 0,
 1,
 0,
 1,
 0,
 0,
 0,
 1,
 0]]

example = pd.DataFrame(data,

columns=['Idade (anos)', 'Pressão sanguínea em repouso (mm Hg)',
       'Colesterol (mm/dl)', 'Açúcar no sangue em jejum',
       'Frequência cardíaca máxima alcançada', 'Oldpeak', 'Sexo_F', 'Sexo_M',
       'Tipo de dor no peito_ASY', 'Tipo de dor no peito_ATA',
       'Tipo de dor no peito_NAP', 'Tipo de dor no peito_TA',
       'Resultados de eletrocardiograma em repouso_LVH',
       'Resultados de eletrocardiograma em repouso_Normal',
       'Resultados de eletrocardiograma em repouso_ST',
       'Angina induzida por exercício_N', 'Angina induzida por exercício_Y',
       'Inclinação do segmento ST de pico do exercício_Down',
       'Inclinação do segmento ST de pico do exercício_Flat',
       'Inclinação do segmento ST de pico do exercício_Up', 'Diagnóstico'])



model = pickle.load(open('knn_model.sav', 'rb'))
ro_scaler = pickle.load(open('ro_scaler.sav', 'rb'))

example = example.drop(['Diagnóstico'], axis=1)
X_valid = ro_scaler.transform(example)

print(model.predict(X_valid))

