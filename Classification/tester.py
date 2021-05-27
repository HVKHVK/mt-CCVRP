import joblib
import pandas as pd

joblib_file = "RFC.pkl"
joblib_model = joblib.load("models/"+joblib_file)

print(joblib_model)
data = pd.DataFrame([[9, 5, 420, 15]])
predict = joblib_model.predict(data)
print(predict)