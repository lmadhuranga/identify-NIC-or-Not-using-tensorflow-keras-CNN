import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from lib import nicPredictor
from utils import config, helpers

appConfig = config.app 

app = Flask(__name__)   

# ********************************* Route Sections **********************************

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():    
    if request.method == 'POST':
        try:
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and helpers.allowed_file(file.filename, config.app['allowedExtentions']):
                filename = secure_filename(helpers.getUniqueName(file.filename)) 
                file.save(os.path.join(appConfig['uploadPath'], filename))                
                # predict using imge
                return jsonify(nicPredictor.runPredict(filename))
            else:
                print('These files only alow', appConfig)
                
        except Exception as e:
            print(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)