# app.py
import os
import time
import json
import logging
import torch
from flask import Flask, render_template, request, jsonify, session
# Compatible werkzeug version issue
try:
    from werkzeug.urls import url_quote
except ImportError:
    from werkzeug.utils import url_quote  # Position in the new version
except ImportError:
    from urllib.parse import quote as url_quote
from Core.models import DocumentQASystem
from Core.document_loader import load_document
from Core.auto_test import perform_auto_test, visualize_results
from Utils.utils import export_to_excel, generate_improved_prompt, save_feedback_data

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session encryption

# Examples of global QA systems
qa_system = DocumentQASystem()

# Configuration log
logging.basicConfig(level=logging.INFO)

# Routing: Home
@app.route('/')
def index():
    return render_template('index.html', models=list(qa_system.llm_registry.keys()))

# Routing: handling Q&A requests
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    model_name = data.get('model', qa_system.current_model)
    
    # Ensure that the chosen model is valid
    if model_name not in qa_system.llm_registry:
        return jsonify({'error': f'Invalid model, available options: {list(qa_system.llm_registry.keys())}'}), 400
    
    # If the model is not the current model, switch
    if model_name != qa_system.current_model:
        qa_system.current_model = model_name
        qa_system._release_model_resources()
        qa_system.conversation_chain = None
    
    try:
        start_time = time.time()
        
        # Processing queries
        if qa_system.qa_chain:  # documentation Q&A model
            result = qa_system.qa_chain({"query": query_text})
            response = f"{result['result']}\nSource: {result['source_documents'][0].metadata['source']}"
        else:  # General conversation mode
            if not qa_system.conversation_chain:
                from langchain.chains import ConversationChain
                qa_system.conversation_chain = ConversationChain(
                    llm=qa_system.llm_registry[qa_system.current_model],
                    memory=qa_system.memory
                )
            response = qa_system.conversation_chain.predict(input=query_text)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Save to session for feedback
        session['last_query'] = query_text
        session['last_response'] = response
        session['last_model'] = model_name
        
        return jsonify({
            'response': response,
            'model': model_name.upper(),
            'latency': f"{latency:.2f}s"
        })
    
    except Exception as e:
        logging.error(f"Process error:{str(e)}")
        return jsonify({'error': f"Process error:{str(e)}"}), 500

# Routing: handling file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded files
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    # Loading Documents
    if load_document(qa_system, file_path):
        return jsonify({'success': True, 'message': 'Document loaded successfully'})
    else:
        return jsonify({'error': 'Document loading failed'}), 500

# Routing: Processing Feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback_type = data.get('type')  # 'like' or 'dislike'
    reason = data.get('reason', '')  # Reasons for dissatisfaction
    
    # Get last query and response
    last_query = session.get('last_query')
    last_response = session.get('last_response')
    last_model = session.get('last_model')
    
    if not last_query or not last_response:
        return jsonify({'error': 'No previous conversation found to evaluate'}), 400
    
    if feedback_type == 'like':
        # Preserving Positive Feedback
        save_feedback_data(last_query, last_response, last_model, "like")
        return jsonify({'success': True, 'message': 'Thank you for your feedback! We will continue to maintain this quality of answers.'})
    
    elif feedback_type == 'dislike':
        # Generate improved prompts
        improved_prompt = generate_improved_prompt(last_query, last_response, "dislike", reason)
        
        try:
            # Re-generate responses using improved prompts
            model = qa_system.llm_registry[qa_system.current_model]
            combined_prompt = f"{improved_prompt}\n\nOriginal question: {last_query}"
            improved_response = model.invoke(combined_prompt)
            
            # Preservation of feedback and improved responses
            save_feedback_data(last_query, last_response, last_model, "dislike", improved_response, reason)
            
            # Update recent responses in session
            session['last_response'] = improved_response
            
            return jsonify({
                'success': True, 
                'message': 'Thank you for your feedback!', 
                'improved_response': improved_response
            })
            
        except Exception as e:
            logging.error(f"Error generating improved answer: {str(e)}")
            return jsonify({'error': f"Error generating improved answer: {str(e)}"}), 500
    
    return jsonify({'error': 'Invalid feedback type'}), 400

# Routing: a comparison of models
@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': 'Please provide query content'}), 400
    
    results = {}
    for name, model in qa_system.llm_registry.items():
        try:
            # Enforce garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Waiting for the system to stabilize
            time.sleep(1)

            # GPU memory usage at the start of the record
            gpu_memory_before = qa_system.get_gpu_memory_usage()
            
            start_time = time.time()
            if qa_system.qa_chain:  # Check for document uploads
                # Generating document-based answers with qa_chain
                from langchain.chains import RetrievalQA
                qa_system.qa_chain = RetrievalQA.from_chain_type(
                    llm=model,
                    chain_type="stuff",
                    retriever=qa_system.vector_db.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                result = qa_system.qa_chain({"query": query_text})
                response = f"{result['result']}\n📚 Source: {result['source_documents'][0].metadata['source']}"
            else:
                # No document upload, direct model call
                response = model.invoke(query_text)
            end_time = time.time()
            
            # Wait for the system to stabilize before measuring the video memory
            time.sleep(1)
            
            # Enforce garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # GPU memory usage at the end of the record
            gpu_memory_after = qa_system.get_gpu_memory_usage()
            
            # Calculating GPU Memory Changes
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
            
            # Estimating the number of tokens (a simple estimate of about 1.3 tokens per word)
            response_words = len(response.split())
            estimated_tokens = int(response_words * 1.3)
            
            # Calculate the speed of token generation
            tokens_per_second = estimated_tokens / latency if latency > 0 else 0
            
            results[name] = {
                "response": response,
                "latency": f"{latency:.2f}s",
                "tokens": estimated_tokens,
                "tokens_per_second": f"{tokens_per_second:.2f}",
                "response_length": len(response),
                "gpu_memory_diff": gpu_memory_diff
            }
            
            # Release of resources
            del response
            qa_system._release_model_resources()
        except Exception as e:
            results[name] = {"error": str(e)}
    
    # Export results to Excel
    export_file = export_to_excel(results, query_text)
    export_path = os.path.abspath(export_file) if export_file else None
    
    return jsonify({
        'results': results,
        'export_file': export_path
    })

# Routing: automated testing
@app.route('/autotest', methods=['POST'])
def autotest():
    data = request.json
    logic_count = int(data.get('logic_count', 0))
    read_count = int(data.get('read_count', 0))
    math_count = int(data.get('math_count', 0))
    
    if logic_count <= 0 and read_count <= 0 and math_count <= 0:
        return jsonify({'error': 'Please select at least one question type to test'}), 400
    
    try:
        # Execute automated tests
        results, standard_answers = perform_auto_test(qa_system, logic_count, read_count, math_count)
        
        # Export results
        from Core.auto_test import export_to_excel as export_test_results
        export_file = export_test_results(results, standard_answers)
        
        if export_file:
            # Generate visualization charts
            chart_file = visualize_results(export_file)
            return jsonify({
                'success': True,
                'message': 'Test completed.',
                'export_file': os.path.abspath(export_file),
                'chart_file': os.path.abspath(chart_file) if chart_file else None
            })
        else:
            return jsonify({'error': 'Failed to export results'}), 500
    
    except Exception as e:
        logging.error(f"Automatically test errors:{str(e)}")
        return jsonify({'error': f"Automatically test errors:{str(e)}"}), 500

# 将reset_mode路由移到if __name__之前
@app.route('/reset_mode', methods=['POST'])
def reset_mode():
    """从文档问答模式切换回普通对话模式"""
    if qa_system.qa_chain:
        qa_system.qa_chain = None
        qa_system.vector_db = None
        qa_system.conversation_chain = None  # 确保重新初始化对话链
        return jsonify({'success': True, 'message': 'Return to conversation mode successfully.'})
    else:
        return jsonify({'success': True, 'message': 'Already in conversation mode.'})

# Launch the application
if __name__ == '__main__':
    # Ensure upload and feedback catalogs exist
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./feedback_data", exist_ok=True)
    
    # Starting a Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
