# main.py
import os
import time
import psutil
import Core
import logging
import torch
from langchain.chains import RetrievalQA
from Core.models import DocumentQASystem
from Core.document_loader import load_document
from Core.auto_test import perform_auto_test, visualize_results
from Utils.utils import print_help, export_to_excel, generate_improved_prompt, save_feedback_data

# 全局变量，用于存储最近的查询和响应
last_query = None
last_response = None
last_model = None

def process_command(command: str, qa_system: DocumentQASystem) -> bool:
    """处理系统命令"""
    global last_query, last_response, last_model
    
    command = command.strip().lower()

    if command.startswith("/switch"):
        model_name = command.split()[-1] if len(command.split()) > 1 else ""
        if model_name in qa_system.llm_registry:
            qa_system.current_model = model_name
            qa_system._release_model_resources()
            qa_system.conversation_chain = None
            print(f"已切换到 {model_name.upper()} 模型")
            return True
        else:
            print(f"❌ 无效模型，可用选项：{list(qa_system.llm_registry.keys())}")
            return False

    # 处理用户反馈 - 赞同
    elif command == "/like":
        if not last_query or not last_response:
            print("❌ 没有找到可以评价的上一次对话")
            return False
        
        print("✅ 感谢您的反馈！我们会继续保持这样的回答质量。")
        save_feedback_data(last_query, last_response, last_model, "like")
        return True
    
    # 处理用户反馈 - 不赞同
    elif command.startswith("/dislike"):
        if not last_query or not last_response:
            print("❌ 没有找到可以评价的上一次对话")
            return False
        
        # 提取反馈原因
        reason = command[8:].strip() if len(command) > 8 else None
        if not reason:
            reason = input("请简单描述您不满意的原因: ")
        
        print("🔄 正在根据您的反馈生成改进的回答...")
        
        # 生成改进的提示词
        improved_prompt = generate_improved_prompt(last_query, last_response, "dislike", reason)
        
        # 使用改进的提示词重新生成回答
        try:
            model = qa_system.llm_registry[qa_system.current_model]
            
            # 组合原始查询和改进提示词
            combined_prompt = f"{improved_prompt}\n\n原始问题: {last_query}"
            
            # 重新生成回答
            improved_response = model.invoke(combined_prompt)
            
            print(f"\n{qa_system.current_model.upper()} (改进后):", improved_response)
            
            # 保存反馈和改进的回答
            save_feedback_data(last_query, last_response, last_model, "dislike", 
                              improved_response, reason)
            
            # 更新最近的响应
            last_response = improved_response
            
        except Exception as e:
            error_msg = f"❌ 生成改进回答时出错：{str(e)}"
            logging.error(error_msg)
            print(error_msg)
        
        return True

    # main.py 中的 /compare 命令处理部分
    elif command.startswith("/compare"):
        query_part = command[8:].strip()
        if not query_part:
            print("请输入多行查询（输入空行结束）：")
            lines = []
            while True:
                line = input().rstrip()
                if line == "":
                    break
                lines.append(line)
            query = "\n".join(lines)
        else:
            query = query_part

        if not query:
            print("请提供查询内容，格式：/compare [查询内容]")
            return False

        print(f"\n正在比较各模型的响应...")
        results = {}
        for name, model in qa_system.llm_registry.items():
            try:
                # 强制执行垃圾回收
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 等待系统稳定
                time.sleep(2)
                
                # 记录开始时的内存和显存使用
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # 转换为MB
                
                # 记录开始时的GPU显存使用
                gpu_memory_before = qa_system.get_gpu_memory_usage()
                
                start_time = time.time()
                if qa_system.qa_chain:  # 检查是否有文档上传
                    # 使用 qa_chain 生成基于文档的回答
                    qa_system.qa_chain = RetrievalQA.from_chain_type(
                        llm=model,
                        chain_type="stuff",
                        retriever=qa_system.vector_db.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    result = qa_system.qa_chain({"query": query})
                    response = f"{result['result']}\n📚 来源：{result['source_documents'][0].metadata['source']}"
                else:
                    # 没有文档上传，直接调用模型
                    response = model.invoke(query)
                end_time = time.time()
                
                # 记录结束时的内存使用
                memory_after = process.memory_info().rss / 1024 / 1024  # 转换为MB
                memory_usage = memory_after - memory_before
                
                # 等待系统稳定后再测量显存
                time.sleep(2)
                
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
                        
                        # 只计算显存使用变化，不再计算利用率
                        memory_diff = round(after_stats["used_memory_mb"] - before_stats["used_memory_mb"], 2)
                        
                        # 获取峰值显存使用
                        peak_memory = after_stats.get("peak_memory_mb", after_stats["used_memory_mb"])
                        
                        gpu_memory_diff[device_id] = {
                            "device_name": before_stats["device_name"],
                            "memory_diff_mb": memory_diff,
                            "peak_memory_mb": peak_memory
                        }
                # 删除以下与显存利用率相关的代码
                # util_before = round(before_stats["utilization_percent"], 2)
                # util_after = round(after_stats["utilization_percent"], 2)
                
                # 计算一致性指标 - 显存使用与利用率变化是否一致
                # memory_change_direction = 1 if memory_diff > 0 else (-1 if memory_diff < 0 else 0)
                # util_change_direction = 1 if (util_after - util_before) > 0 else (-1 if (util_after - util_before) < 0 else 0)
                # consistency = "一致" if memory_change_direction == util_change_direction else "不一致"
                
                # gpu_memory_diff[device_id] = {
                #     "device_name": before_stats["device_name"],
                #     "memory_diff_mb": memory_diff,
                #     "utilization_before": util_before,
                #     "utilization_after": util_after,
                #     "utilization_diff": round(util_after - util_before, 2),
                #     "consistency": consistency
                # }
                
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
                    "memory_usage": f"{memory_usage:.2f}",
                    "gpu_memory_diff": gpu_memory_diff
                }
                
                # 释放资源
                del response
                qa_system._release_model_resources()
            except Exception as e:
                print(f"{name.upper()} 调用出错：{str(e)}")

        print("\n比较结果：")
        for model_name, data in results.items():
            print(f"{model_name.upper()}:")
            print(f"延迟：{data['latency']}")
            print(f"估计令牌数：{data['tokens']}")
            print(f"令牌生成速度：{data['tokens_per_second']} tokens/s")
            print(f"响应长度：{data['response_length']} 字符")
            print(f"内存使用：{data['memory_usage']} MB")
            
            # 显示GPU显存使用情况
            if "gpu_memory_diff" in data and data["gpu_memory_diff"]:
                print("GPU显存使用情况:")
                for device_id, gpu_stats in data["gpu_memory_diff"].items():
                    print(f"  {gpu_stats['device_name']}:")
                    print(f"    显存使用变化: {gpu_stats['memory_diff_mb']} MB")
                    if "peak_memory_mb" in gpu_stats:
                        print(f"    显存使用峰值: {gpu_stats['peak_memory_mb']} MB")
            else:
                print("GPU显存信息: 不可用")
            
            print(f"响应预览：{data['response'][:100]}...\n")

        export_file = export_to_excel(results, query)
        if export_file:
            print(f"\n已导出结果至：{os.path.abspath(export_file)}")
        else:
            print("❌ \n导出结果失败")
        return True

    # /autotest - 自动测试命令
    elif command.startswith("/autotest"):
        # 询问测试数量
        logic_count = int(input("测试的逻辑题数量："))
        read_count = int(input("测试的阅读理解题数量："))
        math_count = int(input("测试的数学题数量："))

        # 调用相关函数加载问题并进行测试
        print(f"开始自动测试 {logic_count} 道逻辑题，{read_count} 道阅读理解题，{math_count} 道数学题...")
        results, standard_answers = perform_auto_test(qa_system, logic_count, read_count, math_count)

        # 导出并可视化结果
        export_file = Core.auto_test.export_to_excel(results, standard_answers)
        if export_file:
            print(f"\n测试结果已导出至：{os.path.abspath(export_file)}")
            visualize_results(export_file)  # 生成可视化图表
        else:
            print("❌ 导出结果失败")

        return True

    elif command == "/help":
        print_help()
        return True

    elif command == "/upload":
        file_path = input("📂 请输入文件路径：").strip()
        if not os.path.exists(file_path):
            print("❌ 文件不存在")
            return False
        if load_document(qa_system, file_path):
            print("📄  文档加载成功")
            return True
        print("❌ 文档加载失败")
        return False

    else:
        print("❌ 未知命令，输入/help查看帮助")
        return False

def process_query(query: str, qa_system: DocumentQASystem, current_model: str) -> None:
    """处理用户查询"""
    global last_query, last_response, last_model
    
    try:
        if qa_system.qa_chain:
            result = qa_system.qa_chain({"query": query})
            response = f"{result['result']}\n来源：{result['source_documents'][0].metadata['source']}"
        else:
            if not qa_system.conversation_chain:
                from langchain.chains import ConversationChain
                qa_system.conversation_chain = ConversationChain(
                    llm=qa_system.llm_registry[qa_system.current_model],
                    memory=qa_system.memory
                )
            response = qa_system.conversation_chain.predict(input=query)

        print(f"\n{current_model.upper()}:", response)
        
        # 保存最近的查询和响应，以便反馈
        last_query = query
        last_response = response
        last_model = current_model
        
        # 提示用户可以提供反馈
        print("\n💬 您可以使用 /like 表示赞同，或 /dislike [原因] 表示不赞同")

    except Exception as e:
        error_msg = f"❌ 处理错误：{str(e)}"
        logging.error(error_msg)
        print(error_msg)

def main():
    qa_system = DocumentQASystem()
    print_help()
    current_model = qa_system.current_model
    
    # 确保反馈目录存在
    os.makedirs("./feedback_data", exist_ok=True)

    while True:
        try:
            # 初始化输入收集
            user_input = []
            print("\nYou: (输入内容，连按两次回车提交)")

            # 多行输入循环
            while True:
                line = input().strip()

                # 退出指令处理
                if line.lower() in ["exit", "quit","/exit","/quit"]:
                    print("👋  再见！")
                    return

                # 命令立即执行
                if line.startswith("/"):
                    process_command(line, qa_system)
                    current_model = qa_system.current_model
                    break

                # 空行表示提交输入
                if not line:
                    if user_input:
                        full_query = "\n".join(user_input)
                        print("🤖 Model思考中......")
                        process_query(full_query, qa_system, current_model)
                    user_input = []
                    break

                user_input.append(line)

        except KeyboardInterrupt:
            print("\n输入中断，输入 exit 退出程序")
        except Exception as e:
            logging.error(f"系统错误：{str(e)}")
            print("❌ 发生意外错误，请重新尝试")

if __name__ == "__main__":
    main()