import os
import pickle
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

class SentimentAnalyzerBase:
    def __init__(self):
        self.word2vec_model = None
        self.model = None
        self.scaler = None
    
    def _get_text_vector(self, text):
        """通用文本向量化方法"""
        words = [w for w in jieba.cut(text) if w.strip()]
        vectors = []
        
        for word in words:
            if word in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[word])
            else:
                # 处理OOV词语，用零向量代替
                vectors.append(np.zeros(self.word2vec_model.vector_size))
        
        if not vectors:
            return None
        
        return np.mean(vectors, axis=0).reshape(1, -1)
    
    def _get_probabilities(self, features):
        """通用概率计算方法"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(features)[0]
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(features)[0]
            prob = 1 / (1 + np.exp(-abs(decision)))
            return np.array([1 - prob, prob]) if decision > 0 else np.array([prob, 1 - prob])
        else:
            return np.array([0.4, 0.6])  # 默认概率

class SentimentAnalyzerSVM(SentimentAnalyzerBase):
    def load_models(self, model_dir):
        """加载SVM模型相关文件"""
        try:
            self.word2vec_model = Word2Vec.load(os.path.join(model_dir, 'word2vec_svm.model'))
            with open(os.path.join(model_dir, 'svm_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            return True
        except Exception as e:
            print(f"加载SVM模型失败: {e}")
            return False

    def predict_sentiment(self, text):
        """SVM情感预测"""
        if not all([self.word2vec_model, self.model, self.scaler]):
            return {'error': '模型未正确加载'}
        
        text_vector = self._get_text_vector(text)
        if text_vector is None:
            return {'error': '无法提取文本特征'}
        
        try:
            scaled_vector = self.scaler.transform(text_vector)
            probabilities = self._get_probabilities(scaled_vector)
            prediction = np.argmax(probabilities)
            
            return {
                'text': text,
                'sentiment': 'positive' if prediction == 1 else 'negative',
                'probability': float(probabilities[prediction]),
                'probabilities': {
                    'positive': float(probabilities[1]),
                    'negative': float(probabilities[0])
                }
            }
        except Exception as e:
            print(f"SVM预测异常: {e}")
            return {
                'text': text,
                'sentiment': 'negative',
                'probability': 0.5,
                'error': str(e)
            }

class SentimentAnalyzerXGB(SentimentAnalyzerBase):
    def load_models(self, model_dir):
        """加载XGBoost模型相关文件"""
        try:
            self.word2vec_model = Word2Vec.load(os.path.join(model_dir, 'word2vec.model'))
            with open(os.path.join(model_dir, 'xgb_model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            return True
        except Exception as e:
            print(f"加载XGBoost模型失败: {e}")
            return False

    def predict_sentiment(self, text):
        """XGBoost情感预测"""
        if not all([self.word2vec_model, self.model]):
            return {'error': '模型未正确加载'}
        
        text_vector = self._get_text_vector(text)
        if text_vector is None:
            return {'error': '无法提取文本特征'}
        
        try:
            probabilities = self._get_probabilities(text_vector)
            prediction = np.argmax(probabilities)
            
            return {
                'text': text,
                'sentiment': 'positive' if prediction == 1 else 'negative',
                'probability': float(probabilities[prediction]),
                'probabilities': {
                    'positive': float(probabilities[1]),
                    'negative': float(probabilities[0])
                }
            }
        except Exception as e:
            print(f"XGB预测异常: {e}")
            return {
                'text': text,
                'sentiment': 'negative',
                'probability': 0.5,
                'error': str(e)
            }