from src import db, login_manager
from src import bcrypt
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(length=30), nullable=False, unique=True)
    email_address = db.Column(db.String(length=50), nullable=False, unique=True)
    password = db.Column(db.String(length=60), nullable=False)
    image_embedding = db.Column(db.PickleType, nullable=True)  # For storing face recognition data
    
    # You can add any additional methods or properties here
    def __repr__(self):
        return f'User {self.username}'


