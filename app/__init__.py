# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'd48334684ef950f3b704b584d5477eb2'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/sentiment_app'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'

    # 注册蓝图等
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    with app.app_context():
        from .models import User  # 延迟导入模型
        db.create_all()

    return app

# 用户加载回调函数
@login_manager.user_loader
def load_user(user_id):
    from .models import User  # 延迟导入以避免循环依赖
    return User.query.get(int(user_id))