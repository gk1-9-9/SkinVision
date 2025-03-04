from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, HiddenField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError
from src.models import User


class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email_address = StringField('Email Address', validators=[DataRequired(), Email()])
    password1 = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=30)])
    password2 = PasswordField('Confirm Password', validators=[DataRequired(), Length(min=6, max=30)])
    image_data = HiddenField('Image Data', validators=[DataRequired()])  # Hidden field for the captured image data
    submit = SubmitField('Register')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    image_data = HiddenField('Image Data')  # Hidden field for the captured image data
    submit = SubmitField('Login')

class CheckForm(FlaskForm):
    image_data = HiddenField('Image Data1')  # Hidden field for the captured image data
    submit = SubmitField('image')

class CheckJaundiceForm(FlaskForm):
    image_data = HiddenField('Image Data1')  # Hidden field for the captured image data
    submit = SubmitField('image')

class mailForm(FlaskForm): # Hidden field for the captured image data
    pass



