from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
import os
from werkzeug.utils import secure_filename
from flask import session



app = Flask(__name__)
model_path = "resnet_50.h5"
model = load_model(model_path)

# Secret key for session security (replace with a strong and unique string)
app.secret_key = 'your_very_strong_and_secret_key_here123!#'

# To Load images
#app.config['STATIC_FOLDER'] = 'test' 

# List of your two classes
class_labels = ['Non_Violence', 'Violence']

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
  filename = session.get('filename', None)  # Handle case if no filename in session
  return render_template('dashboard.html', image_filename=filename)

@app.route('/dashboard/map', methods=['GET'])
def map_view():
    return render_template('map.html')

@app.route('/', methods=['POST'])
def predict():
    file = request.files['imagefile']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = load_img(file_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)

        # Assuming the model output is the predicted class index (0 or 1)
        predicted_class_index = yhat.argmax()

        # Get the corresponding label from the list of class labels
        predicted_label = class_labels[predicted_class_index]

        location = request.form.get('location')  # Get location information from the form

        # Add location information to the predicted label
        classification = '%s (%.2f%%): %s' % (predicted_label, yhat[0, predicted_class_index] * 100, location)
        session['filename'] = filename
        return render_template('index.html', prediction=classification, image_filename=filename)
    
        #return render_template('dashboard.html', prediction=classification, image_filename=filename)



if __name__ == '__main__':
    app.run(port=3000, debug=True)