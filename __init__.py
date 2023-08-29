from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'

    from model import model

    app.register_blueprint(model, url_prefix='/')

    return app