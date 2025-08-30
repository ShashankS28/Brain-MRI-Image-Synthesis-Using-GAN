from flask import Flask, request, render_template,jsonify
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras.models import Sequential
import os
app = Flask(__name__)

# Load the pre-trained generator model with custom object scope
try:
    generator = load_model("generator_model.h5", compile=False)
except Exception as e:
    print(f"Error loading generator model: {e}")
    generator = None
UPLOAD_FOLDER = 'static/uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key="1234567"

@app.route('/',methods=['GET', 'POST'])
def login():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        
                # If account exists in accounts table in out database
        if username=="admin" and password=="admin":
            # Create session data, we can access this data in other routes
            # Redirect to home page
            return render_template('index.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)


@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/generate")
def generate():
    # Render the HTML file
    return render_template("input.html")

@app.route("/generate_images", methods=["POST"])
def generate_images():
    try:
        if generator is None:
            return jsonify({"error": "Generator model not loaded. Please check model file."}), 500
            
        # Parse user input from the front end
        data = request.get_json()
        num_images = int(data.get("num_images", 1))
        noise_dim = int(data.get("noise_dim", 100))

        # Generate random noise
        noise = np.random.normal(0, 1, (num_images, noise_dim))

        # Generate images using the generator model
        generated_images = generator.predict(noise)

        # Convert generated images to base64 strings
        image_list = []
        for img in generated_images:
            img = (img * 127.5 + 127.5).astype("uint8")  # Rescale pixel values to [0, 255]
            img = Image.fromarray(img.squeeze(), mode="L")  # Convert to grayscale image
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_list.append(f"data:image/png;base64,{img_str}")

        return jsonify({"images": image_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])
class_names = ['glioma_tumor', 'meningioma_tumor','no_tumor','pituitary_tumor']
class_names1 = ['Non Valid', 'valid']
img_height = 224
img_width = 224
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/home1')
def home1():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('home.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    print(file)
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
        ])

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        model.load_weights("BrainValid.h5")

        test_data_path = path

        img = keras.preprocessing.image.load_img(
            test_data_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        msg="This Image is Not Valid"
        if class_names1[np.argmax(score)]=="Non Valid":
            return render_template('result.html', aclass=msg,filename=filename,res=2)
        else:
            num_classes = 4
            model = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
            ])
            model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

            model.load_weights("BrainTumour.h5")

            test_data_path = path

            img = keras.preprocessing.image.load_img(
                test_data_path, target_size=(img_height, img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(np.argmax(score), 100 * np.max(score))
            )
            print(np.argmax(score))
            return render_template('result.html', filename=filename,aclass=class_names[np.argmax(score)],ascore=100 * np.max(score),res=1)


        
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('thankyou'))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)