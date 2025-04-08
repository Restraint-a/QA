// main.js - Intelligent Q&A System Frontend Interaction Script

document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const chatTab = document.getElementById('chat-tab');
    const uploadTab = document.getElementById('upload-tab');
    const compareTab = document.getElementById('compare-tab');
    const autotestTab = document.getElementById('autotest-tab');
    
    const chatSection = document.getElementById('chat-section');
    const uploadSection = document.getElementById('upload-section');
    const compareSection = document.getElementById('compare-section');
    const autotestSection = document.getElementById('autotest-section');
    
    const modelSelector = document.getElementById('model-selector');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    
    const docQueryInput = document.getElementById('doc-query-input');
    const docSendBtn = document.getElementById('doc-send-btn');
    const docChatMessages = document.getElementById('doc-chat-messages');
    
    const uploadForm = document.getElementById('upload-form');
    const documentFile = document.getElementById('document-file');
    
    const compareQuery = document.getElementById('compare-query');
    const compareBtn = document.getElementById('compare-btn');
    const compareResults = document.getElementById('compare-results');
    
    const autotestForm = document.getElementById('autotest-form');
    const logicCount = document.getElementById('logic-count');
    const readCount = document.getElementById('read-count');
    const mathCount = document.getElementById('math-count');
    const testResults = document.getElementById('test-results');
    
    const loadingOverlay = document.getElementById('loading-overlay');
    const feedbackModal = new bootstrap.Modal(document.getElementById('feedback-modal'));
    const feedbackReason = document.getElementById('feedback-reason');
    const submitDislike = document.getElementById('submit-dislike');
    
    // 当前活动的消息元素（用于反馈）
    let activeMessageElement = null;
    
    // 切换标签页
    function switchTab(tab, section) {
        // 移除所有标签页的active类
        [chatTab, uploadTab, compareTab, autotestTab].forEach(t => t.classList.remove('active'));
        // 隐藏所有内容区域
        [chatSection, uploadSection, compareSection, autotestSection].forEach(s => s.style.display = 'none');
        
        // 激活选中的标签页和内容区域
        tab.classList.add('active');
        section.style.display = 'block';
    }
    
    // 绑定标签页切换事件
    chatTab.addEventListener('click', () => switchTab(chatTab, chatSection));
    uploadTab.addEventListener('click', () => switchTab(uploadTab, uploadSection));
    compareTab.addEventListener('click', () => switchTab(compareTab, compareSection));
    autotestTab.addEventListener('click', () => switchTab(autotestTab, autotestSection));
    
    // 显示加载中遮罩
    function showLoading() {
        loadingOverlay.style.display = 'flex';
    }
    
    // 隐藏加载中遮罩
    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }
    
    // 格式化时间
    function formatTime() {
        const now = new Date();
        return now.toLocaleTimeString();
    }
    
    // 添加消息到聊天区域
    function addMessage(container, content, isUser, model = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        let headerText = isUser ? 'You' : model;
        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header';
        headerDiv.textContent = headerText;
        messageDiv.appendChild(headerDiv);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        messageDiv.appendChild(contentDiv);
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = formatTime();
        messageDiv.appendChild(timeDiv);
        
        // 如果是机器人消息，添加反馈按钮
        if (!isUser) {
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'feedback-buttons';
            
            const likeBtn = document.createElement('button');
            likeBtn.className = 'feedback-btn';
            likeBtn.innerHTML = '<i class="bi bi-hand-thumbs-up"></i> Agree';
            likeBtn.addEventListener('click', () => handleFeedback('like', messageDiv));
            
            const dislikeBtn = document.createElement('button');
            dislikeBtn.className = 'feedback-btn';
            dislikeBtn.innerHTML = '<i class="bi bi-hand-thumbs-down"></i> Disagree';
            dislikeBtn.addEventListener('click', () => {
                activeMessageElement = messageDiv;
                feedbackModal.show();
            });
            
            feedbackDiv.appendChild(likeBtn);
            feedbackDiv.appendChild(dislikeBtn);
            messageDiv.appendChild(feedbackDiv);
        }
        
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
        return messageDiv;
    }
    
    // 处理用户查询
    async function processQuery(input, messagesContainer, isDocMode = false) {
        const query = input.value.trim();
        if (!query) return;
        
        // 添加用户消息
        addMessage(messagesContainer, query, true);
        input.value = '';
        
        // 显示加载中
        showLoading();
        
        try {
            // 发送请求到后端
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    model: modelSelector.value
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // 添加机器人回复
                addMessage(messagesContainer, data.response, false, data.model);
            } else {
                // 显示错误
                addMessage(messagesContainer, `Error: ${data.error}`, false, 'System');
            }
        } catch (error) {
            console.error('Request Error:', error);
            addMessage(messagesContainer, `Network Error: ${error.message}`, false, 'System');
        } finally {
            hideLoading();
        }
    }
    
    // 处理反馈
    async function handleFeedback(type, messageElement) {
        showLoading();
        
        try {
            let data = { type: type };
            
            // 如果是不赞同，添加原因
            if (type === 'dislike') {
                data.reason = feedbackReason.value.trim();
                feedbackModal.hide();
                feedbackReason.value = '';
            }
            
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                // 显示成功消息
                if (type === 'like') {
                    alert(result.message);
                } else if (type === 'dislike' && result.improved_response) {
                    // 如果有改进的回答，更新消息内容
                    const contentDiv = messageElement.querySelector('.message-content');
                    if (contentDiv) {
                        contentDiv.textContent = result.improved_response;
                        
                        // 添加更新标记
                        const headerDiv = messageElement.querySelector('.message-header');
                        if (headerDiv) {
                            headerDiv.textContent += ' (Improved)';
                        }
                        
                        // 更新时间
                        const timeDiv = messageElement.querySelector('.message-time');
                        if (timeDiv) {
                            timeDiv.textContent = formatTime();
                        }
                        
                        // 移除反馈按钮
                        const feedbackDiv = messageElement.querySelector('.feedback-buttons');
                        if (feedbackDiv) {
                            messageElement.removeChild(feedbackDiv);
                        }
                    }
                    
                    alert('Thank you for your feedback! Answer has been improved.');
                }
            } else {
                alert(`Feedback submission error: ${result.error}`);
            }
        } catch (error) {
            console.error('Feedback submission error:', error);
            alert(`Feedback submission error: ${error.message}`);
        } finally {
            hideLoading();
        }
    }
    
    // 处理文件上传
    async function handleFileUpload(event) {
        event.preventDefault();
        
        if (!documentFile.files[0]) {
            alert('Please select file');
            return;
        }
        
        showLoading();
        
        const formData = new FormData();
        formData.append('file', documentFile.files[0]);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                alert(result.message);
                // 在清空文件选择前保存文件名
                const fileName = documentFile.files[0].name;
                // 清空文件选择
                documentFile.value = '';
                // 添加系统消息
                addMessage(docChatMessages, `The document "${fileName}" has been successfully loaded and you can now ask questions about the document.`, false, 'System');
            } else {
                alert(`Upload Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Upload Error:', error);
            alert(`Upload Error: ${error.message}`);
        } finally {
            hideLoading();
        }
    }
    
    // 处理模型比较
    async function handleCompare() {
        const query = compareQuery.value.trim();
        if (!query) {
            alert('Please enter your query');
            return;
        }
        
        showLoading();
        compareResults.innerHTML = '';
        
        try {
            const response = await fetch('/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // 创建结果标题
                const titleDiv = document.createElement('h4');
                titleDiv.textContent = 'Comparing Result:';
                compareResults.appendChild(titleDiv);
                
                // 创建查询信息
                const queryDiv = document.createElement('div');
                queryDiv.className = 'alert alert-info';
                queryDiv.textContent = `Query: ${query}`;
                compareResults.appendChild(queryDiv);
                
                // 为每个模型创建结果卡片
                for (const [modelName, result] of Object.entries(data.results)) {
                    const cardDiv = document.createElement('div');
                    cardDiv.className = 'model-result-card';
                    
                    const headerDiv = document.createElement('div');
                    headerDiv.className = 'model-result-header';
                    headerDiv.textContent = modelName.toUpperCase();
                    cardDiv.appendChild(headerDiv);
                    
                    const bodyDiv = document.createElement('div');
                    bodyDiv.className = 'model-result-body';
                    
                    // 如果有错误
                    if (result.error) {
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'alert alert-danger';
                        errorDiv.textContent = `错误: ${result.error}`;
                        bodyDiv.appendChild(errorDiv);
                    } else {
                        // 显示指标
                        const metricsDiv = document.createElement('div');
                        metricsDiv.className = 'model-metrics';
                        
                        // 添加各项指标
                        const metrics = [
                            { label: 'Latency', value: result.latency },
                            { label: 'Estimated number of tokens', value: result.tokens },
                            { label: 'Token Consumption Rate', value: `${result.tokens_per_second} tokens/s` },
                            { label: 'Response Length', value: `${result.response_length} 字符` }
                        ];
                        
                        metrics.forEach(metric => {
                            const metricDiv = document.createElement('div');
                            metricDiv.className = 'metric';
                            
                            const labelDiv = document.createElement('div');
                            labelDiv.className = 'metric-label';
                            labelDiv.textContent = metric.label;
                            metricDiv.appendChild(labelDiv);
                            
                            const valueDiv = document.createElement('div');
                            valueDiv.className = 'metric-value';
                            valueDiv.textContent = metric.value;
                            metricDiv.appendChild(valueDiv);
                            
                            metricsDiv.appendChild(metricDiv);
                        });
                        
                        bodyDiv.appendChild(metricsDiv);
                        
                        // 显示GPU信息
                        if (result.gpu_memory_diff && Object.keys(result.gpu_memory_diff).length > 0) {
                            const gpuDiv = document.createElement('div');
                            gpuDiv.className = 'gpu-info';
                            
                            for (const [deviceId, gpuStats] of Object.entries(result.gpu_memory_diff)) {
                                const deviceDiv = document.createElement('div');
                                deviceDiv.className = 'metric';
                                
                                const deviceLabel = document.createElement('div');
                                deviceLabel.className = 'metric-label';
                                deviceLabel.textContent = `GPU: ${gpuStats.device_name}`;
                                deviceDiv.appendChild(deviceLabel);
                                
                                const memoryDiff = document.createElement('div');
                                memoryDiff.className = 'metric-value';
                                memoryDiff.textContent = `GPU Memory Change: ${gpuStats.memory_diff_mb} MB`;
                                deviceDiv.appendChild(memoryDiff);
                                
                                if (gpuStats.peak_memory_mb) {
                                    const peakMemory = document.createElement('div');
                                    peakMemory.className = 'metric-value';
                                    peakMemory.textContent = `GPU Memory Peak: ${gpuStats.peak_memory_mb} MB`;
                                    deviceDiv.appendChild(peakMemory);
                                }
                                
                                gpuDiv.appendChild(deviceDiv);
                            }
                            
                            bodyDiv.appendChild(gpuDiv);
                        }
                        
                        // 显示响应内容
                        const responseTitle = document.createElement('h6');
                        responseTitle.className = 'mt-3 mb-2';
                        responseTitle.textContent = 'Response:';
                        bodyDiv.appendChild(responseTitle);
                        
                        const responseDiv = document.createElement('div');
                        responseDiv.className = 'model-response';
                        responseDiv.textContent = result.response;
                        bodyDiv.appendChild(responseDiv);
                    }
                    
                    cardDiv.appendChild(bodyDiv);
                    compareResults.appendChild(cardDiv);
                }
                
                // 显示导出信息
                if (data.export_file) {
                    const exportDiv = document.createElement('div');
                    exportDiv.className = 'alert alert-success mt-3';
                    exportDiv.textContent = `Results have been exported to: ${data.export_file}`;
                    compareResults.appendChild(exportDiv);
                }
            } else {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'alert alert-danger';
                errorDiv.textContent = `Comparison Error: ${data.error}`;
                compareResults.appendChild(errorDiv);
            }
        } catch (error) {
            console.error('Comparison Error:', error);
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger';
            errorDiv.textContent = `Comparison Error: ${error.message}`;
            compareResults.appendChild(errorDiv);
        } finally {
            hideLoading();
        }
    }
    
    // 处理自动测试
    async function handleAutoTest(event) {
        event.preventDefault();
        
        const logicValue = parseInt(logicCount.value) || 0;
        const readValue = parseInt(readCount.value) || 0;
        const mathValue = parseInt(mathCount.value) || 0;
        
        if (logicValue <= 0 && readValue <= 0 && mathValue <= 0) {
            alert('Please select at least one question type to test');
            return;
        }
        
        showLoading();
        testResults.innerHTML = '';
        
        try {
            const response = await fetch('/autotest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    logic_count: logicValue,
                    read_count: readValue,
                    math_count: mathValue
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // 创建结果标题
                const titleDiv = document.createElement('h4');
                titleDiv.textContent = 'Test Result';
                testResults.appendChild(titleDiv);
                
                // 创建测试信息
                const infoDiv = document.createElement('div');
                infoDiv.className = 'alert alert-info';
                infoDiv.textContent = `Completed tests: ${logicValue} logic questions, ${readValue} reading comprehension questions, ${mathValue} math questions.`;
                testResults.appendChild(infoDiv);
                
                // 如果有图表文件，显示图表
                if (data.chart_file) {
                    const chartImg = document.createElement('img');
                    chartImg.src = data.chart_file + '?t=' + new Date().getTime(); // 添加时间戳防止缓存
                    chartImg.className = 'img-fluid mt-3 mb-3';
                    chartImg.alt = 'Test Result Chart';
                    testResults.appendChild(chartImg);
                }
                
                // 显示导出信息
                if (data.export_file) {
                    const exportDiv = document.createElement('div');
                    exportDiv.className = 'alert alert-success mt-3';
                    exportDiv.textContent = `Detailed results have been exported to. ${data.export_file}`;
                    testResults.appendChild(exportDiv);
                }
            } else {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'alert alert-danger';
                errorDiv.textContent = `Test error: ${data.error}`;
                testResults.appendChild(errorDiv);
            }
        } catch (error) {
            console.error('Test error:', error);
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger';
            errorDiv.textContent = `Test error: ${error.message}`;
            testResults.appendChild(errorDiv);
        } finally {
            hideLoading();
        }
    }
    
    // 绑定事件处理函数
    sendBtn.addEventListener('click', () => processQuery(queryInput, chatMessages));
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            processQuery(queryInput, chatMessages);
        }
    });
    
    docSendBtn.addEventListener('click', () => processQuery(docQueryInput, docChatMessages, true));
    docQueryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            processQuery(docQueryInput, docChatMessages, true);
        }
    });
    
    uploadForm.addEventListener('submit', handleFileUpload);
    compareBtn.addEventListener('click', handleCompare);
    autotestForm.addEventListener('submit', handleAutoTest);
    
    submitDislike.addEventListener('click', () => {
        if (activeMessageElement) {
            handleFeedback('dislike', activeMessageElement);
        }
    });
    
    // 添加欢迎消息
    addMessage(chatMessages, 'Welcome to the Intelligent Q&A System!Please enter your question.', false, 'System');
    addMessage(docChatMessages, 'Please upload the document first, then you can ask questions about the document.', false, 'System');
});