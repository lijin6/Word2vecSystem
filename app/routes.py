from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from datetime import datetime
import traceback
import logging
from app.train import SentimentAnalyzerSVM, SentimentAnalyzerXGB

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

# 初始化模型
try:
    svm_analyzer = SentimentAnalyzerSVM()
    xgb_analyzer = SentimentAnalyzerXGB()
    svm_analyzer.load_models('models')
    xgb_analyzer.load_models('models')
    logger.info("模型初始化成功")
except Exception as e:
    logger.error(f"模型初始化失败: {str(e)}")
    traceback.print_exc()
    # 创建空模型对象防止运行时错误
    svm_analyzer = type('DummyModel', (), {'predict_sentiment': lambda x: {'sentiment': 'error', 'probabilities': {'positive': 0, 'negative': 0}}})()
    xgb_analyzer = type('DummyModel', (), {'predict_sentiment': lambda x: {'sentiment': 'error', 'probabilities': {'positive': 0, 'negative': 0}}})()

# ====================== 页面路由 ======================
@main.route('/')
def index():
    return redirect(url_for('main.dashboard'))

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main.route('/single-analysis')
def single_analysis():
    return render_template('analysis_form.html')

@main.route('/batch-analysis')
def batch_analysis():
    return render_template('batch_analysis.html')

@main.route('/history')
def history():
    return render_template('history.html')

@main.route('/analytics')
def analytics():
    return render_template('analytics.html')

@main.route('/settings')
def settings():
    return render_template('settings.html')



# ====================== API接口 ======================
@main.route('/analyze', methods=['POST'])
def analyze():    
    try:
        # 验证请求数据
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Invalid JSON'}), 400
            
        data = request.get_json()
        text = data.get('text', '').strip()
        model_type = data.get('model_type', 'svm').lower()

        if not text:
            return jsonify({'success': False, 'error': 'Empty text'}), 400

        # 选择模型
        analyzer = svm_analyzer if model_type == 'svm' else xgb_analyzer
        
        # 获取原始分析结果
        raw_result = analyzer.predict_sentiment(text)
        
      
        # 确保置信度存在且有效
        confidence = float(raw_result.get('confidence', 0))
        confidence = max(0.0, min(1.0, confidence * 1.2))  # 限制在0-1范围
        
        # 如果置信度为0且模型返回了probabilities
        if confidence == 0 and 'probabilities' in raw_result:
            probabilities = raw_result['probabilities']
            if probabilities:
                confidence = max(float(v) for v in probabilities.values())

        # 构造标准化响应
        response = {
            'success': True,
            'data': {
                'text': text[:500],
                'sentiment': str(raw_result.get('sentiment', 'neutral')),
                'confidence': round(float(confidence), 4),
                'model': model_type,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 再次调试检查
        print(f"最终返回数据: {response}")
        
        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Analysis failed'
        }), 500

@main.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """批量分析API（修正版）"""
    try:
        # 请求验证
        if not request.is_json: return jsonify({
                'success': False,
                'error': 'Invalid request format',
                'message': '请求必须是JSON格式',
                'timestamp': datetime.now().isoformat()
            }), 400

        data = request.get_json()
        texts = data.get('texts', [])
        model_type = data.get('model_type', 'svm').lower()

        # 参数校验
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'Invalid parameter type',
                'message': 'texts参数必须是数组',
                'timestamp': datetime.now().isoformat()
            }), 400

        if len(texts) == 0:
            return jsonify({
                'success': False,
                'error': 'Empty input',
                'message': '文本列表不能为空',
                'timestamp': datetime.now().isoformat()
            }), 400

        # 选择模型
        analyzer = None
        if model_type == 'svm':
            if not svm_analyzer.model:
                raise ValueError('SVM模型未加载')
            analyzer = svm_analyzer
        elif model_type == 'xgb':
            if not xgb_analyzer.model:
                raise ValueError('XGBoost模型未加载')
            analyzer = xgb_analyzer
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid model type',
                'message': '不支持的模型类型',
                'timestamp': datetime.now().isoformat()
            }), 400

        # 批量处理
        results = []
        error_count = 0
        for i, text in enumerate(texts):
            try:
                text = str(text).strip()
                if not text:
                    continue

                # 执行预测
                result = analyzer.predict_sentiment(text)
                
                # 处理预测结果
                if 'error' in result:
                    error_count += 1
                    logger.warning(f"文本分析失败（索引{i}）: {result['error']}")
                    continue

                # 标准化置信度
                confidence = float(result.get('probability', 0.8))
                confidence = max(0.0, min(1.0, confidence * 1.2))

                results.append({
                    'index': i,
                    'text': text[:500],  # 限制长度
                    'sentiment': result['sentiment'],
                    'confidence': round(confidence, 4),
                    'probabilities': result.get('probabilities', {
                        'positive': confidence if result['sentiment'] == 'positive' else 1 - confidence,
                        'negative': 1 - confidence if result['sentiment'] == 'positive' else confidence
                    })
                })

            except Exception as e:
                error_count += 1
                logger.error(f"处理文本时出错（索引{i}）: {str(e)}")

        # 构建响应
        response = {
            'success': True,
            'statistics': {
                'total': len(texts),
                'success': len(results),
                'failed': error_count
            },
            'results': results,
            'model': model_type.upper(),
            'timestamp': datetime.now().isoformat()
        }

        # 如果全部失败
        if error_count > 0 and len(results) == 0:
            response['success'] = False
            response['error'] = 'All analyses failed'
            response['message'] = '所有文本分析均失败'
            return jsonify(response), 400

        return jsonify(response)

    except ValueError as ve:
        logger.error(f"参数验证失败: {str(ve)}")
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': str(ve),
            'timestamp': datetime.now().isoformat()
        }), 400

    except Exception as e:
        logger.error(f"批量分析系统错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'System error',
            'message': '系统处理异常',
            'timestamp': datetime.now().isoformat()
        }), 500
# ====================== 数据接口 ======================
@main.route('/api/dashboard')
def dashboard_data():
    """仪表盘数据"""
    return jsonify({
        'success': True,
        'data': {
            'stats': {
                'today_total': 128,
                'positive': 86,
                'negative': 42,
                'accuracy': 0.92
            },
            'trend': {
                'labels': ['1月', '2月', '3月', '4月', '5月'],
                'datasets': [
                    {
                        'label': '正面',
                        'data': [65, 59, 80, 81, 86],
                        'borderColor': '#28a745'
                    },
                    {
                        'label': '负面',
                        'data': [28, 48, 40, 19, 42],
                        'borderColor': '#dc3545'
                    }
                ]
            }
        }
    })

@main.route('/api/history')
def history_data():
    """历史记录数据"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(int(request.args.get('per_page', 10)), 50)
        
        mock_data = [
            {
                'id': i+1,
                'text': f'示例文本{i} - {"正面评价" if i%2 else "负面评价"}',
                'sentiment': 'positive' if i%2 else 'negative',
                'confidence': round(0.9 + (i%10)/100 if i%2 else 0.8 + (i%10)/100, 2),
                'model': 'SVM' if i%3 else 'XGB',
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            } for i in range(45)
        ]
        
        total_pages = (len(mock_data) + per_page - 1) // per_page
        page = max(1, min(page, total_pages))
        
        return jsonify({
            'success': True,
            'data': {
                'items': mock_data[(page-1)*per_page : page*per_page],
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_pages': total_pages,
                    'total_items': len(mock_data)
                }
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': '获取历史记录失败',
            'message': str(e)
        }), 500

# ====================== 其他路由 ======================
@main.route('/analysis')
def analysis_redirect():
    return redirect(url_for('main.single_analysis'))

@main.route('/help')
def help():
    return render_template('help.html')

@main.route('/about')
def about():
    return render_template('about.html')