# utils.py
import os
import logging
import pandas as pd
from datetime import datetime


DEFAULT_PATH = "./compare_output"
FEEDBACK_PATH = "./feedback_data"

def print_help():
    print("""
🔧 系统指令手册：
    /switch [mistral|qwen|deepseek|gemma3]  - 切换大语言模型
    /compare [问题]         - 对比模型性能
    /help                  - 显示帮助信息
    /upload                - 上传文档进入问答模式
    /autotest              - 自动测试
    /like                  - 赞同上一次回答
    /dislike [原因]         - 不赞同上一次回答，可提供原因
    /exit/quit              - 退出系统
""")

def export_to_excel(results, query):
    try:
        safe_query = query.replace("\n", " ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.xlsx"

        # 组装数据
        data = []
        for model_name, result in results.items():
            # 基本信息
            row = {
                "Model": model_name.upper(),
                "Query": safe_query,
                "Latency": result["latency"],
                "Tokens": result.get("tokens", "N/A"),
                "Tokens/Second": result.get("tokens_per_second", "N/A"),
                "Response Length": result.get("response_length", "N/A"),
                "Memory Usage (MB)": result.get("memory_usage", "N/A"),
            }
            
            # 添加GPU显存信息 - 只保留显存使用峰值和变化
            if "gpu_memory_diff" in result and result["gpu_memory_diff"]:
                for device_id, gpu_stats in result["gpu_memory_diff"].items():
                    # 只保留显存使用变化和峰值
                    row[f"Memory_Diff_MB"] = gpu_stats["memory_diff_mb"]
                    if "peak_memory_mb" in gpu_stats:
                        row[f"Peak_Memory_MB"] = gpu_stats["peak_memory_mb"]
                
            
            # 添加响应内容
            row["Response"] = result["response"]
            data.append(row)
            
        # 创建DataFrame并导出
        df = pd.DataFrame(data)
        
        # 确保输出目录存在
        os.makedirs(DEFAULT_PATH, exist_ok=True)
        output_path = os.path.join(DEFAULT_PATH, filename)
        
        # 导出到Excel
        df.to_excel(output_path, index=False)
        return output_path
    except Exception as e:
        logging.error(f"导出失败: {str(e)}")
        return None

def generate_improved_prompt(query, response, feedback, reason=None):
    """根据用户反馈生成改进的提示词"""
    if feedback == "like":
        # 如果用户喜欢这个回答，记录这种模式以便将来使用
        return None  # 不需要修改提示词
    
    # 用户不喜欢回答时，生成改进的提示词
    improved_prompt = f"""
        我之前的回答不够好。原问题是: 
        "{query}"

        我的回答是:
        "{response}"

        用户反馈: 不满意
        """
    
    if reason:
        improved_prompt += f"原因: {reason}\n"
    
    improved_prompt += """
        请根据以上反馈，提供一个更好的回答。注意:
        1. 保持回答的准确性和相关性
        2. 提供更详细的解释和例子
        3. 确保回答逻辑清晰，易于理解
        """
    
    return improved_prompt

def save_feedback_data(query, response, model_name, feedback, improved_response=None, reason=None):
    """保存用户反馈数据"""
    try:
        # 确保反馈目录存在
        os.makedirs(FEEDBACK_PATH, exist_ok=True)
        
        # 准备反馈数据
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feedback_data = {
            "Timestamp": [timestamp],
            "Model": [model_name.upper()],
            "Query": [query],
            "Original Response": [response],
            "Feedback": [feedback],
            "Reason": [reason if reason else ""],
            "Improved Response": [improved_response if improved_response else ""]
        }
        
        # 创建或追加到反馈文件
        feedback_file = os.path.join(FEEDBACK_PATH, "user_feedback.xlsx")
        
        # 如果文件存在，追加数据
        if os.path.exists(feedback_file):
            existing_df = pd.read_excel(feedback_file)
            new_df = pd.DataFrame(feedback_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(feedback_file, index=False)
        else:
            # 创建新文件
            pd.DataFrame(feedback_data).to_excel(feedback_file, index=False)
            
        return feedback_file
    except Exception as e:
        logging.error(f"保存反馈失败: {str(e)}")
        return None