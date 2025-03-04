from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///byteCraft.db'
app.config['SECRET_KEY'] = 'll'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login_page"
login_manager.login_message_category = "info"
app.config['MAIL_SERVER'] = 'smtp.office365.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] ='o@o.com'
app.config['MAIL_PASSWORD'] = 'hello'
mail = Mail(app)


@login_manager.user_loader
def load_user(user_id):
    from src.models import User  # Import here to avoid circular imports
    return User.query.get(int(user_id))


from src import routes
