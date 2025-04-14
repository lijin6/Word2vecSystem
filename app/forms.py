# app/forms.py
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired

class SentimentForm(FlaskForm):
    text = TextAreaField('请输入文本', validators=[DataRequired()])
    model_type = SelectField('选择模型', choices=[('svm', 'SVM'), ('xgb', 'XGBoost')])
    submit = SubmitField('提交分析')
