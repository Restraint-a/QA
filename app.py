# app.py
import os
import time
import json
import logging
import torch
from flask import Flask, render_template, request, jsonify, session
# 兼容werkzeug版本问题
try:
    from werkzeug.urls import url_quote
except ImportError:
    from werkzeug.utils import url_quote  # 新版本中的位置
except ImportError:
    from urllib.parse import quote as url_quote  # 最后的备选方案
from Core.models import DocumentQASystem
from Core.document_loader import load_document
from Core.auto_test import perform_auto_test, visualize_results
from Utils.utils import export_to_excel, generate_improved_prompt, save_feedback_data

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于session加密

# 全局QA系统实例
qa_system = DocumentQASystem()

# 配置日志
logging.basicConfig(level=logging.INFO)

# 路由：主页
@app.route('/')
def index():
    return render_template('index.html', models=list(qa_system.llm_registry.keys()))

# 路由：处理问答请求
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    model_name = data.get('model', qa_system.current_model)
    
    # 确保选择的模型有效
    if model_name not in qa_system.llm_registry:
        return jsonify({'error': f'无效模型，可用选项：{list(qa_system.llm_registry.keys())}'}), 400
    
    # 如果模型不是当前模型，则切换
    if model_name != qa_system.current_model:
        qa_system.current_model = model_name
        qa_system._release_model_resources()
        qa_system.conversation_chain = None
    
    try:
        start_time = time.time()
        
        # 处理查询
        if qa_system.qa_chain:  # 文档问答模式
            result = qa_system.qa_chain({"query": query_text})
            response = f"{result['result']}\n来源：{result['source_documents'][0].metadata['source']}"
        else:  # 普通对话模式
            if not qa_system.conversation_chain:
                from langchain.chains import ConversationChain
                qa_system.conversation_chain = ConversationChain(
                    llm=qa_system.llm_registry[qa_system.current_model],
                    memory=qa_system.memory
                )
            response = qa_system.conversation_chain.predict(input=query_text)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # 保存到session以便反馈
        session['last_query'] = query_text
        session['last_response'] = response
        session['last_model'] = model_name
        
        return jsonify({
            'response': response,
            'model': model_name.upper(),
            'latency': f"{latency:.2f}s"
        })
    
    except Exception as e:
        logging.error(f"处理错误：{str(e)}")
        return jsonify({'error': f"处理错误：{str(e)}"}), 500

# 路由：处理文件上传
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 保存上传的文件
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    # 加载文档
    if load_document(qa_system, file_path):
        return jsonify({'success': True, 'message': '文档加载成功'})
    else:
        return jsonify({'error': '文档加载失败'}), 500

# 路由：处理反馈
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback_type = data.get('type')  # 'like' 或 'dislike'
    reason = data.get('reason', '')  # 不满意的原因
    
    # 获取上次的查询和响应
    last_query = session.get('last_query')
    last_response = session.get('last_response')
    last_model = session.get('last_model')
    
    if not last_query or not last_response:
        return jsonify({'error': '没有找到可以评价的上一次对话'}), 400
    
    if feedback_type == 'like':
        # 保存正面反馈
        save_feedback_data(last_query, last_response, last_model, "like")
        return jsonify({'success': True, 'message': '感谢您的反馈！我们会继续保持这样的回答质量。'})
    
    elif feedback_type == 'dislike':
        # 生成改进的提示词
        improved_prompt = generate_improved_prompt(last_query, last_response, "dislike", reason)
        
        try:
            # 使用改进的提示词重新生成回答
            model = qa_system.llm_registry[qa_system.current_model]
            combined_prompt = f"{improved_prompt}\n\n原始问题: {last_query}"
            improved_response = model.invoke(combined_prompt)
            
            # 保存反馈和改进的回答
            save_feedback_data(last_query, last_response, last_model, "dislike", improved_response, reason)
            
            # 更新session中的最近响应
            session['last_response'] = improved_response
            
            return jsonify({
                'success': True, 
                'message': '感谢您的反馈！', 
                'improved_response': improved_response
            })
            
        except Exception as e:
            logging.error(f"生成改进回答时出错：{str(e)}")
            return jsonify({'error': f"生成改进回答时出错：{str(e)}"}), 500
    
    return jsonify({'error': '无效的反馈类型'}), 400

# 路由：模型比较
@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': '请提供查询内容'}), 400
    
    results = {}
    for name, model in qa_system.llm_registry.items():
        try:
            # 强制执行垃圾回收
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 等待系统稳定
            time.sleep(1)
            
            # 记录开始时的GPU显存使用
            gpu_memory_before = qa_system.get_gpu_memory_usage()
            
            start_time = time.time()
            if qa_system.qa_chain:  # 检查是否有文档上传
                # 使用qa_chain生成基于文档的回答
                from langchain.chains import RetrievalQA
                qa_system.qa_chain = RetrievalQA.from_chain_type(
                    llm=model,
                    chain_type="stuff",
                    retriever=qa_system.vector_db.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                result = qa_system.qa_chain({"query": query_text})
                response = f"{result['result']}\n📚 来源：{result['source_documents'][0].metadata['source']}"
            else:
                # 没有文档上传，直接调用模型
                response = model.invoke(query_text)
            end_time = time.time()
            
            # 等待系统稳定后再测量显存
            time.sleep(1)
            
            # 强制执行垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 记录结束时的GPU显存使用
            gpu_memory_after = qa_system.get_gpu_memory_usage()
            
            # 计算GPU显存变化
            gpu_memory_diff = {}
            if gpu_memory_before["available"] and gpu_memory_after["available"]:
                for device_id, before_stats in gpu_memory_before["devices"].items():
                    after_stats = gpu_memory_after["devices"][device_id]
                    memory_diff = round(after_stats["used_memory_mb"] - before_stats["used_memory_mb"], 2)
                    peak_memory = after_stats.get("peak_memory_mb", after_stats["used_memory_mb"])
                    
                    gpu_memory_diff[device_id] = {
                        "device_name": before_stats["device_name"],
                        "memory_diff_mb": memory_diff,
                        "peak_memory_mb": peak_memory
                    }
            
            latency = end_time - start_time
            
            # 估算令牌数量 (简单估计，每个单词约1.3个令牌)
            response_words = len(response.split())
            estimated_tokens = int(response_words * 1.3)
            
            # 计算令牌生成速度
            tokens_per_second = estimated_tokens / latency if latency > 0 else 0
            
            results[name] = {
                "response": response,
                "latency": f"{latency:.2f}s",
                "tokens": estimated_tokens,
                "tokens_per_second": f"{tokens_per_second:.2f}",
                "response_length": len(response),
                "gpu_memory_diff": gpu_memory_diff
            }
            
            # 释放资源
            del response
            qa_system._release_model_resources()
        except Exception as e:
            results[name] = {"error": str(e)}
    
    # 导出结果到Excel
    export_file = export_to_excel(results, query_text)
    export_path = os.path.abspath(export_file) if export_file else None
    
    return jsonify({
        'results': results,
        'export_file': export_path
    })

# 路由：自动测试
@app.route('/autotest', methods=['POST'])
def autotest():
    data = request.json
    logic_count = int(data.get('logic_count', 0))
    read_count = int(data.get('read_count', 0))
    math_count = int(data.get('math_count', 0))
    
    if logic_count <= 0 and read_count <= 0 and math_count <= 0:
        return jsonify({'error': '请至少选择一种题型进行测试'}), 400
    
    try:
        # 执行自动测试
        results, standard_answers = perform_auto_test(qa_system, logic_count, read_count, math_count)
        
        # 导出结果
        from Core.auto_test import export_to_excel as export_test_results
        export_file = export_test_results(results, standard_answers)
        
        if export_file:
            # 生成可视化图表
            chart_file = visualize_results(export_file)
            return jsonify({
                'success': True,
                'message': '测试完成',
                'export_file': os.path.abspath(export_file),
                'chart_file': os.path.abspath(chart_file) if chart_file else None
            })
        else:
            return jsonify({'error': '导出结果失败'}), 500
    
    except Exception as e:
        logging.error(f"自动测试错误：{str(e)}")
        return jsonify({'error': f"自动测试错误：{str(e)}"}), 500

# 启动应用
if __name__ == '__main__':
    # 确保上传和反馈目录存在
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./feedback_data", exist_ok=True)
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)