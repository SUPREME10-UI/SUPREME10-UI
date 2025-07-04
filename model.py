from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# ChatSession model to track individual user sessions
class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100), nullable=True)  # Store the user's name, if available
    start_time = db.Column(db.DateTime, default=datetime.utcnow)  # Session start time
    end_time = db.Column(db.DateTime, nullable=True)  # Session end time (if available)
    active = db.Column(db.Boolean, default=True)  # Track if the session is still active
    chat_logs = db.relationship('ChatLog', backref='session', lazy=True)  # Relationship with chat logs
    
    def __init__(self, user_name=None):
        self.user_name = user_name

    def end_session(self):
        self.end_time = datetime.utcnow()
        self.active = False

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_name={self.user_name}, active={self.active})>"

# ChatLog model to store individual chat exchanges
class ChatLog(db.Model):
    __tablename__ = 'chat_logs'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    user_message = db.Column(db.String(500), nullable=False)  # Store user's message
    bot_response = db.Column(db.String(500), nullable=False)  # Store bot's response
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp of the message exchange

    def __init__(self, session_id, user_message, bot_response):
        self.session_id = session_id
        self.user_message = user_message
        self.bot_response = bot_response

    def __repr__(self):
        return f"<ChatLog(id={self.id}, session_id={self.session_id}, user_message={self.user_message}, bot_response={self.bot_response})>"

