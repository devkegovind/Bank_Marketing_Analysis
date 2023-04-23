from flask import Flask, request, render_template, jsonify
from bank.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(
            job = request.form.get('job'),
            marital = request.form.get('marital'),
            education = request.form.get('education'),
            default = request.form.get('default'),
            housing = request.form.get('housing'),
            loan = request.form.get('housing'),
            contact = request.form.get('contact'),
            month = request.form.get('month'),
            poutcome = request.form.get('poutcome'),
            age = int(request.form.get('age')),
            balance = int(request.form.get('balance')),
            day = int(request.form.get('day')),
            campaign = int(request.form.get('campaign')),
            pdays = int(request.form.get('pdays')),
            previous = int(request.form.get('previous'))
        )
        
        final_new_data = data.get_data_as_dataframe()
        
        predict_pipeline = PredictPipeline()
        
        pred = predict_pipeline.predict(final_new_data)
        
        results = int(pred[0])
        
        return render_template('result.html', final_result = results)
        
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)