

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        gravity = float(request.form['gravity'])
        ph = float(request.form['ph'])
        osmo = float(request.form['osmo'])
        cond = float(request.form['cond'])
        urea = float(request.form['urea'])
        calc = float(request.form['calc'])


        val = np.array([gravity, ph, osmo,cond, urea,calc])
        datain = [np.array(val)]

        scalar_path = os.path.join('models','ScalarMinMax')
        scalar = pickle.load(open(scalar_path, 'rb'))

        # final_features = [np.array(val)]
        model_path = os.path.join('models','model_batuginjal_NB.pkl')
        model = pickle.load(open(model_path, 'rb'))
        # res = model.predict(final_features)

        final_features = scalar.transform(datain)
        res = model.predict(final_features)

        if res[0] == 1:
            target = "Tidak Terdapat Batu Ginjal"
        else:
            target = "Terdapat Batu Ginjal"

        return render_template('index.html', result=target)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)