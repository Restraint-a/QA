# utils.py
import pandas as pd
from datetime import datetime
import logging
import os

DEFAULT_PATH = "./compare_output"
def print_help():
    print("""
🔧 系统指令手册：
    /switch [mistral|qwen|deepseek]  - 切换大语言模型
    /compare [问题]         - 对比模型性能
    /help                  - 显示帮助信息
    /upload                - 上传文档进入问答模式
    exit/quit              - 退出系统
""")

def export_to_excel(results, query):
    try:
        safe_query = query.replace("\n", " ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.xlsx"

        # 组装数据
        data = []
        for model_name, result in results.items():
            row = {
                "Model": model_name.upper(),
                "Query": safe_query,
                "Latency": result["latency"],
                "Response Preview": result["response"]
            }
            data.append(row)

        # 构造 DataFrame
        df = pd.DataFrame(data)
        df = df[["Model", "Query", "Latency", "Response Preview"]]

        filename = os.path.join(DEFAULT_PATH, filename)

        # 导出 Excel 文件
        df.to_excel(filename, index=False, engine="openpyxl")
        return filename
    except Exception as e:
        logging.error(f"导出失败: {str(e)}")
        return None