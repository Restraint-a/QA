import os
import re
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from Core.models import DocumentQASystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_questions_and_answers(question_type: str, question_set_dir: str = "./questions_set") -> Dict[str, dict]:
    """
    加载题目和标准答案
    :param question_type: 题型（logic/math）
    :param question_set_dir: 题库根目录
    :return: 结构化题目数据 {qid: {"question": ..., "answer": ...}}
    """
    data = {}
    base_path = os.path.join(question_set_dir, question_type)

    if question_type == "math":
        # 加载数学题
        with open(os.path.join(base_path, "math.txt"), "r", encoding="utf-8") as f:
            content = f.read()
        questions = re.split(r"\n\d+\.", content)  # 按题号分割
        questions = [q.strip() for q in questions if q.strip()]

        # 加载数学答案
        with open(os.path.join(base_path, "math_answer.txt"), "r", encoding="utf-8") as f:
            answers = re.findall(r"(\d+)【Answer】(\w+)", f.read())
            answer_map = {qid: option.upper() for qid, option in answers}

        # 组合数据
        for idx, q in enumerate(questions, 1):
            data[f"math_{idx}"] = {
                "question": q,
                "answer_option": answer_map.get(str(idx), ""),
                "answer_analysis": ""  # 数学题无分析文本
            }

    elif question_type in ["logic", "read"]:
        # 修改：改进逻辑/阅读理解题的加载方式
        for filename in os.listdir(base_path):
            if filename.endswith(".txt") and "answer" not in filename:
                # 从文件名中提取题号，例如从 "logic_3.txt" 提取 "3"
                file_parts = filename.split(".")
                if len(file_parts) > 0:
                    file_base = file_parts[0]  # 例如 "logic_3"
                    qid_parts = file_base.split("_")
                    if len(qid_parts) > 1:
                        qid = qid_parts[1]  # 提取题号，例如 "3"

                        # 读取题目文件
                        with open(os.path.join(base_path, filename), "r", encoding="utf-8") as f:
                            content = f.read()

                            # 提取最后一个问题（通常是题目的最后部分）
                            question_parts = content.split("\n\n")
                            for i in range(len(question_parts)-1, -1, -1):
                                if question_parts[i].strip():
                                    question = question_parts[i].strip()
                                    break
                            else:
                                question = content.strip()

                        # 加载答案文件
                        answer_file = os.path.join(base_path, f"{file_base}_answer.txt")
                        if os.path.exists(answer_file):
                            with open(answer_file, "r", encoding="utf-8") as f:
                                answer_content = f.read().strip()

                                # 匹配题号和答案，支持多种格式
                                # 例如 "3.【Answer】B" 或 "3【Answer】B"
                                answer_match = re.search(r"(\d+)\.?【Answer】(\w+)([\s\S]*)", answer_content)
                                #print(answer_match)
                                if answer_match:
                                    answer_num = answer_match.group(1)
                                    option = answer_match.group(2).upper() if answer_match.group(2) else ""
                                    analysis = answer_match.group(3).strip() if answer_match.group(3) else ""

                                    # 确保题号匹配
                                    if answer_num == qid:
                                        data[f"{question_type}_{qid}"] = {
                                            "question": question,
                                            "answer_option": option,
                                            "answer_analysis": analysis
                                        }
                                        print(f"成功加载 {question_type}_{qid} 题目")
                                    else:
                                        print(f"警告：{answer_file} 中的题号 {answer_num} 与文件名中的题号 {qid} 不匹配")
                                else:
                                    print(f"警告：无法从 {answer_file} 中提取答案")
                        else:
                            print(f"警告：答案文件 {answer_file} 不存在")

    return data

def get_model_response(qa_system: DocumentQASystem, question: str) -> Tuple[str, float]:
    """
    获取模型回答并计算延迟
    :return: (response, latency)
    """
    start_time = time.time()
    try:
        response = qa_system.llm_registry[qa_system.current_model].invoke(question)
    except Exception as e:
        response = f"Error: {str(e)}"
    latency = time.time() - start_time
    return response, latency


def evaluate_response(model_answer: str, item: dict, q_type: str) -> dict:
    """
    计算accuracy和relevance
    """
    metrics = {
        "accuracy": 0.0,
        "relevance": 0.0,
        "completed_rate": 0.0
    }

    # 提取模型答案选项
    option_match = re.search(r"\b([A-D])\b", model_answer)
    model_option = option_match.group(1).upper() if option_match else ""
    print(f"处理题目: {item.get('question', '')[:30]}... 模型选项: {model_option}")

    # 准确率评估（仅逻辑和数学题）
    if q_type in ["logic", "math"]:
        correct_option = item.get("answer_option", "").upper()
        metrics["accuracy"] = 1.0 if model_option == correct_option else 0.0

    # 相关性评估（逻辑和阅读题）
    if q_type in ["logic", "read"]:
        metrics["relevance"] = compute_similarity(model_answer, item["answer_analysis"])

    # 完成度评估（逻辑题专用）
    if q_type == "logic":
        metrics["completed_rate"] = min(max(metrics["relevance"] * 1.5, 0.0), 1.0)  # 相关性越高完成度越高

    return metrics

# 相似度评分函数
# 方案1：基于 TF-IDF 的余弦相似度（轻量级）
def compute_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度（范围 0~1）
    """
    # 预处理文本（可选）
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()

    # 空文本处理
    if not text1 or not text2:
        return 0.0

    # 创建 TF-IDF 向量器
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(max(similarity, 0.0), 2)  # 确保非负


# ---------- 或 ---------- #

# 方案2：基于 Sentence-BERT 的语义相似度（高精度，需GPU加速）
# from sentence_transformers import SentenceTransformer
#
# # 加载预训练模型（首次运行需下载）
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#
#
# def compute_similarity(text1: str, text2: str) -> float:
#     """
#     计算语义相似度（范围 0~1）
#     """
#     # 编码文本
#     embeddings = model.encode([text1, text2], convert_to_tensor=True)
#     similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
#     return round(max(similarity, 0.0), 2)

def perform_auto_test(
        qa_system: DocumentQASystem,
        logic_count: int,
        read_count: int,
        math_count: int
) -> Tuple[List[dict], Dict[str, str]]:
    """
    执行自动测试并返回结果
    """
    results = []
    standard_answers = {}

    # 加载题目
    logic_data = load_questions_and_answers("logic")
    read_data = load_questions_and_answers("read")
    math_data = load_questions_and_answers("math")
    
    # 打印加载的题目数量
    print(f"加载的逻辑题数量: {len(logic_data)}")
    print(f"加载的阅读题数量: {len(read_data)}")
    print(f"加载的数学题数量: {len(math_data)}")
    
    # 如果请求的题目数量超过了可用题目，调整为可用的最大数量
    logic_count = min(logic_count, len(logic_data))
    read_count = min(read_count, len(read_data))
    math_count = min(math_count, len(math_data))
    
    print(f"实际测试的逻辑题数量: {logic_count}")
    print(f"实际测试的阅读题数量: {read_count}")
    print(f"实际测试的数学题数量: {math_count}")

    # 遍历所有注册的模型
    for model_name in qa_system.llm_registry.keys():
        qa_system.current_model = model_name
        qa_system._release_model_resources()  # 释放前一个模型的资源

        # 测试逻辑题
        for qid, item in list(logic_data.items())[:logic_count]:
            response, latency = get_model_response(qa_system, item["question"])
            metrics = evaluate_response(response, item, "logic")
            results.append({
                "model": model_name,
                "qid": qid,
                "type": "logic",
                "latency": latency,
                **metrics
            })

        # 测试阅读理解题
        for qid, item in list(read_data.items())[:read_count]:
            response, latency = get_model_response(qa_system, item["question"])
            metrics = evaluate_response(response, item, "read")
            results.append({
                "model": model_name,
                "qid": qid,
                "type": "read",
                "latency": latency,
                **metrics
            })

        # 测试数学题
        for qid, item in list(math_data.items())[:math_count]:
            response, latency = get_model_response(qa_system, item["question"])
            metrics = evaluate_response(response, item, "math")
            results.append({
                "model": model_name,
                "qid": qid,
                "type": "math",
                "latency": latency,
                **metrics
            })

    return results, standard_answers


def export_to_excel(results: list, standard_map: dict) -> str:
    """
    导出结果到Excel（保存到/compare_output目录）
    """
    # 确保目录存在
    output_dir = "./compare_output"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    filename = os.path.join(output_dir, f"autotest_{time.strftime('%Y%m%d-%H%M%S')}.xlsx")
    df.to_excel(filename, index=False)
    return filename


def visualize_results(excel_path: str):
    """生成可视化报表"""
    df = pd.read_excel(excel_path)
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体列表
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 创建画布和子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle("Multi-models comparisons", fontsize=16, y=0.95)

    # ================== 准确率对比 ==================
    ax1 = axes[0, 0]
    accuracy_data = df[df['type'].isin(['logic', 'math'])] \
                        .groupby(['model', 'type'])['accuracy'].mean() * 100
    if not accuracy_data.empty:
        accuracy_data.unstack().plot.bar(ax=ax1)
        ax1.set_title('Accuracy Comparison', pad=15)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 110)
        ax1.tick_params(axis='x', rotation=45)
        # 添加数值标签
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.1f%%', padding=3)
    else:
        ax1.text(0.5, 0.5, 'No Accuracy data', ha='center', va='center')

    # ================== 相关性对比 ==================
    ax2 = axes[0, 1]
    relevance_data = df[df['type'].isin(['logic', 'read'])] \
                         .groupby(['model', 'type'])['relevance'].mean() * 100
    if not relevance_data.empty:
        relevance_data.unstack().plot.bar(ax=ax2, color=['#1f77b4', '#ff7f0e'])
        ax2.set_title('Relevance Comparison', pad=15)
        ax2.set_ylabel('Relevance (%)')
        ax2.set_ylim(0, 110)
        ax2.tick_params(axis='x', rotation=45)
        # 添加数值标签
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f%%', padding=3)
    else:
        ax2.text(0.5, 0.5, 'No Relevance data', ha='center', va='center')

    # ================== 完成度对比 ==================
    ax3 = axes[1, 0]
    completion_data = df[df['type'] == 'logic'] \
                          .groupby('model')['completed_rate'].mean() * 100
    if not completion_data.empty:
        bars = completion_data.plot.bar(ax=ax3, color='#2ca02c')
        ax3.set_title('Logic Completion Comparison', pad=15)
        ax3.set_ylabel('completion (%)')
        ax3.set_ylim(0, 110)
        ax3.tick_params(axis='x', rotation=45)
        # 添加数值标签
        ax3.bar_label(bars.containers[0], fmt='%.1f%%', padding=3)
    else:
        ax3.text(0.5, 0.5, 'No Completion data', ha='center', va='center')

    # ================== 延迟对比 ==================
    ax4 = axes[1, 1]
    latency_data = df.groupby(['model', 'type'])['latency'].mean().unstack()
    if not latency_data.empty:
        latency_data.plot.bar(ax=ax4)
        ax4.set_title('Average Latency Comparison', pad=15)
        ax4.set_ylabel('Latency (s)')
        ax4.tick_params(axis='x', rotation=45)
        # 添加数值标签
        for container in ax4.containers:
            ax4.bar_label(container, fmt='%.1f', padding=3)
    else:
        ax4.text(0.5, 0.5, 'No Latency data', ha='center', va='center')

    # 保存图表
    plot_path = excel_path.replace(".xlsx", "_visual.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"可视化图表已保存至：{plot_path}")