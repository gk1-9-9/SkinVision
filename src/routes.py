from src import app, mail
from flask import render_template, redirect, url_for, flash, request, jsonify
from src.models import User
from src.forms import RegisterForm, LoginForm, CheckForm, CheckJaundiceForm, mailForm
from src import db
from flask_login import login_user, logout_user, login_required, current_user
import cv2 
import supervision as sv
import base64
import numpy as np
from werkzeug.security import check_password_hash, generate_password_hash
from src.llm import model, client, model2
from gradio_client import handle_file
from flask_mail import Mail, Message
import datetime

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/succes')
@login_required
def succes_page():
        return render_template('succes.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if request.method == 'POST':
        print("\n=== Form Submission Debug ===")
        print("Headers:", dict(request.headers))
        print("\nForm Data:")
        for field, value in request.form.items():
            if field != 'image_data':
                print(f"{field}: {value}")
            else:
                print(f"image_data length: {len(value) if value else 'NO DATA'}")
        
        print("\nFiles:", request.files)
        
        if not form.validate():
            print("\nForm Validation Errors:")
            for field, errors in form.errors.items():
                print(f"{field}: {errors}")
                flash(f"{field}: {', '.join(errors)}", 'error')
            return render_template('register1.html', form=form)

        try:
            # Check if email already exists
            existing_user = User.query.filter_by(email_address=form.email_address.data).first()
            if existing_user:
                flash("An account with this email address already exists. Please use a different email or login to your existing account.", 'error')
                return render_template('register1.html', form=form)

            # Extract and validate image data
            image_data = form.image_data.data
            if not image_data or not image_data.startswith('data:image'):
                flash("Invalid or missing image data", 'error')
                print("Invalid or missing image data")
                return render_template('register1.html', form=form)

            # Process the valid form data
            username = form.username.data
            email_address = form.email_address.data
            password = form.password1.data
            password_hash = password

            print("\nProcessing valid form submission:")
            print(f"Username: {username}")
            print(f"Email: {email_address}")
            print("Password: [hashed]")
            print(f"Image data present and valid: True")

            try:
                # Process image data
                header, encoded = image_data.split(',', 1)
                binary_image_data = base64.b64decode(encoded)
                nparr = np.frombuffer(binary_image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("Failed to decode image")

                print("Image successfully decoded")
                
                # Create embedding
                img_resized = cv2.resize(img, (100, 100))
                img_embedding = img_resized.flatten()
                
                print("Image embedding created")

                # Create and save user
                user_to_create = User(
                    username=username,
                    email_address=email_address,
                    password=password_hash,
                    image_embedding=img_embedding
                )

                db.session.add(user_to_create)
                db.session.commit()
                print("User successfully added to database")
                
                flash(f"Account created successfully! You are now registered as {username}", category='success')
                return redirect(url_for('login_page'))

            except Exception as e:
                print(f"Image processing error: {str(e)}")
                db.session.rollback()
                flash(f"Error processing image: {str(e)}", category='error')
                return render_template('register1.html', form=form)

        except Exception as e:
            print(f"Registration error: {str(e)}")
            db.session.rollback()
            flash(f"An error occurred during registration: {str(e)}", category='error')
            return render_template('register1.html', form=form)

    return render_template('register1.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if request.method == 'POST':
        try:

            if form.validate_on_submit():
                user = User.query.filter_by(username=form.username.data).first()
                if user and user.password == form.password.data:  # Note: Should use password hashing
                    login_user(user)
                    flash('Logged in successfully!', category='success')
                    return redirect(url_for('home_page'))
                else:
                    flash('Invalid username or password!', category='error')
                    return render_template('login1.html', form=form)
                    

            # Check if this is a face recognition login attempt
            image_data = form.image_data.data
            if image_data and image_data.startswith('data:image'):
                # Skip regular form validation for face recognition
                try:
                    # Process the captured image
                    header, encoded = image_data.split(',', 1)
                    binary_image_data = base64.b64decode(encoded)
                    nparr = np.frombuffer(binary_image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    # save the image 
                    cv2.imwrite('tried_output.png', img)

                    if img is None:
                        raise ValueError("Failed to decode image")

                    # Create embedding for comparison
                    img_resized = cv2.resize(img, (100, 100))
                    login_embedding = img_resized.flatten()

                    # Compare with stored embeddings
                    best_match = None
                    lowest_distance = float('inf')
                    
                    users = User.query.all()
                    for user in users:
                        if user.image_embedding is not None:
                            distance = np.linalg.norm(user.image_embedding - login_embedding)
                            if distance < lowest_distance:
                                lowest_distance = distance
                                best_match = user

                    RECOGNITION_THRESHOLD = 10
                    user = User.query.filter_by(username=form.username.data).first()
                    if user:
                        login_user(user)
                        flash('Logged in successfully!', category='success')
                        return redirect(url_for('home_page'))
                    else:
                        flash('Invalid username or password!', category='error')
                    

                    if best_match and lowest_distance < RECOGNITION_THRESHOLD:
                        login_user(best_match)
                        flash(f'Welcome back, {best_match.username}!', category='success')
                        return redirect(url_for('home_page'))
                    else:
                        flash('Face not recognized. Please try again or use traditional login.', category='error')

                except Exception as e:
                    flash(f'Error during face recognition: {str(e)}', category='error')
                    return render_template('login1.html', form=form)

            # Traditional login validation
            

        except Exception as e:
            flash(f'An error occurred: {str(e)}', category='error')
            
    return render_template('login1.html', form=form)


@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for("home_page"))
strr = ""
@app.route('/check', methods=['GET', 'POST'])
def checkPage():
    form = CheckForm()
    annotated_image_b64 = None
    global strr
    strr = ""  # Initialize variable to hold base64 image
    if request.method == 'POST':
        try:
            image_data = form.image_data.data
            # Decode the base64 image data
            image_data = image_data.split(',')[1]
            image_decoded = base64.b64decode(image_data)
            np_image = np.frombuffer(image_decoded, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            cv2.imwrite('output1.png', image)
            # Perform inference using the model
            results = model.infer(image, conf_thres=0, iou_thres=0.5)[0]
            detections = sv.Detections.from_inference(results)
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # Annotate the image
            annotated_image = bounding_box_annotator.annotate(
                scene=image, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)
            
            cv2.imwrite('output.png', annotated_image)

            

            # Encode the annotated image to base64 for rendering in the template
            _, buffer = cv2.imencode('.png', annotated_image)
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')

            result = client.predict(
                image=handle_file('C:/byteCraft/ByteCraft/output1.png'),
                text_input="write some home remedies and some medications in a paragraph without any special characters and bold text as if you are a doctor.",
                system_prompt="You are an experienced dermatologist and you are rating the severity of the acne in the image.",
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                api_name="/run_example"
            )
            strr = result[0]
            print(strr)
        except Exception as e:
            flash(f"Error loading image: {e}")
            

    return render_template('image_capture.html', form=form, ann=annotated_image_b64, strr = strr)

@app.route('/SendMail', methods=['GET', 'POST'])
def mailPage():
    form = mailForm()
    global strr
    email_sent = False
    if request.method == 'POST' and form.validate_on_submit():
        try:
            # Check if it's an email submission
                if strr:
                    msg = Message(
                        subject='Dermatologist Report - ',
                        sender=app.config['MAIL_USERNAME'],
                        recipients=[current_user.email_address]
                    )
                    
                    msg.body = f"""
Dear {current_user.username},

Here is your dermatological consultation report:

{strr}

Best regards,
Your Dermatology Team
                """
                    mail.send(msg)
                    email_sent = True
                    flash('Email sent successfully!')
                    return redirect(url_for('home_page'))

        except Exception as e:
            flash(f"Error: {str(e)}")

    return render_template('sendMail.html', 
                         form=form, 
                         strr=strr, 
                         email_sent=email_sent)



@app.route('/checkJaundice', methods=['GET', 'POST'])
def checkJaundicePage():
    form = CheckJaundiceForm()
    annotated_image_b64 = None
    strr1 = None  # Initialize variable to hold base64 image
    if request.method == 'POST':
        try:
            image_data = form.image_data.data
            # Decode the base64 image data
            image_data = image_data.split(',')[1]
            image_decoded = base64.b64decode(image_data)
            np_image = np.frombuffer(image_decoded, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            cv2.imwrite('output1.png', image)
            print("hello")
            frame = cv2.imread(r'C:\byteCraft\output1.png')
            if frame is None:
                print("Error: Could not load image.")
            else:
    # Preprocess the image
                img = cv2.resize(frame, (224, 224))  # Resize to the input size of the model
                img = img / 255.0  # Normalize the image
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                
                # Make predictions
                predictions = model2.predict(img)
            print(predictions)
            if predictions[0][0] > 0.9:
                    strr1 = "Jaundice Not Detected"
            else:
                    strr1 = "Jaundice Detected"
            print(strr1)
        except Exception as e:
            flash(f"Error loading image: {e}")
            

    return render_template('jaundice.html', form=form, strr1 = strr1)










