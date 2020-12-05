from flask import *
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')

@app.route("/kidney")
def index():
    return render_template("kidney.html")

def valuepredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if (size == 9):
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #kidney
        if(len(to_predict_list)==9):
            result = valuepredictor(to_predict_list,9)

    if (int(result) == 1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return (render_template("result.html", prediction_text=prediction))

if __name__ == '__main__':
    app.run(debug= True)
