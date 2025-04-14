# app/__init__.py
from flask import Flask
from .routes import main

def create_app():
    app = Flask(__name__)

    # 注册蓝图
    app.register_blueprint(main)

    return app
