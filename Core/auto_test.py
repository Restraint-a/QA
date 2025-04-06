import os
import re
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from Core.models import DocumentQASystem
import matplotlib.pyplot as plt
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
        answer_map = {qid: ans for qid, ans in answers}

        # 组合数据
        for idx, q in enumerate(questions, 1):
            data[f"math_{idx}"] = {"question": q, "answer": answer_map.get(str(idx), "")}

    elif question_type in ["logic", "read"]:
        # 加载逻辑/阅读理解题（假设文件名格式为logic_1.txt）
        for filename in os.listdir(base_path):
            if filename.endswith(".txt") and "answer" not in filename:
                qid = filename.split(".")[0]
                with open(os.path.join(base_path, filename), "r", encoding="utf-8") as f:
                    question = f.read()
                # 加载答案
                answer_file = os.path.join(base_path, f"{qid}_answer.txt")
                with open(answer_file, "r", encoding="utf-8") as f:
                    answers = re.findall(r"(\d+\.\d+)【Answer】(\w+)", f.read())
                answer_map = {aid: ans for aid, ans in answers}
                # 分割子题目（例如1.1, 1.2）
                sub_questions = re.split(r"\n\d+\.\d+\.", question)
                for sub_q in sub_questions[1:]:
                    lines = sub_q.strip().split("\n")
                    aid = lines[0].split()[0]
                    data[f"{qid}_{aid}"] = {
                        "question": "\n".join(lines),
                        "answer": answer_map.get(aid, "")
                    }
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


def evaluate_response(model_answer: str, standard_answer: str) -> dict:
    """
    计算accuracy和relevance
    """
    # 提取模型答案选项
    model_ans = ""
    match = re.search(r"【Answer】\s*(\w+)", model_answer)  # 宽松匹配
    if match:
        model_ans = match.group(1).strip().upper()  # 统一转为大写
    accuracy = 1.0 if model_ans == standard_answer.upper() else 0.0
    relevance = compute_similarity(model_answer, standard_answer)
    return {"accuracy": accuracy, "relevance": relevance}

# 相似度评分函数
def compute_similarity(text1, text2):
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def perform_auto_test(
        qa_system: DocumentQASystem,
        logic_count: int,
        read_count: int,
        math_count: int
) -> Tuple[Dict[str, list], Dict[str, str]]:
    """
    执行自动测试并返回结果
    """
    results = []
    standard_answers = {}

    # 加载题目
    logic_data = load_questions_and_answers("logic")
    read_data = load_questions_and_answers("read")
    math_data = load_questions_and_answers("math")

    # 遍历所有注册的模型
    for model_name in qa_system.llm_registry.keys():
        # 切换到当前模型
        original_model = qa_system.current_model
        qa_system.current_model = model_name
        qa_system._release_model_resources()  # 释放前一个模型的资源

        # 测试逻辑题
        for qid, item in list(logic_data.items())[:logic_count]:
            response, latency = get_model_response(qa_system, item["question"])
            metrics = evaluate_response(response, item["answer"])
            results.append({
                "model": model_name,
                "qid": qid,
                "type": "logic",
                "latency": latency,
                **metrics,
                "completed": 1 if metrics["accuracy"] > 0 else 0
            })
            standard_answers[qid] = item["answer"]

        # 测试阅读理解题
        for qid, item in list(read_data.items())[:read_count]:
            response, latency = get_model_response(qa_system, item["question"])
            metrics = evaluate_response(response, item["answer"])
            results.append({
                "model": model_name,
                "qid": qid,
                "type": "read",
                "latency": latency,
                **metrics,
                "completed": 1 if metrics["accuracy"] > 0 else 0
            })
            standard_answers[qid] = item["answer"]

        # 测试数学题
        for qid, item in list(math_data.items())[:math_count]:
            response, latency = get_model_response(qa_system, item["question"])
            metrics = evaluate_response(response, item["answer"])
            results.append({
                "model": model_name,
                "qid": qid,
                "type": "math",
                "latency": latency,
                **metrics,
                "completed": 1 if metrics["accuracy"] > 0 else 0
            })
            standard_answers[qid] = item["answer"]

    return results, standard_answers


def export_to_excel(results: list, query: str, standard_answers: dict) -> str:
    """
    导出结果到Excel（保存到/compare_output目录）
    """
    # 确保目录存在
    output_dir = "./compare_output"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df["standard_answer"] = df["qid"].map(standard_answers)

    # 生成带时间戳的文件名
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"autotest_compare_{timestamp}.xlsx")

    df.to_excel(filename, index=False)
    return filename


def visualize_results(excel_path: str):
    """
    生成包含四个指标的综合对比图
    """
    df = pd.read_excel(excel_path)

    # 创建画布和子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle("多模型性能综合对比", fontsize=16)

    # 指标列表和子图坐标
    metrics = ["accuracy", "relevance", "latency", "completed"]
    titles = ["准确率对比", "相关性对比", "延迟对比 (秒)", "任务完成率对比"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, metric in enumerate(metrics):
        ax = axes[positions[i][0], positions[i][1]]

        # 按模型和题型分组计算均值
        pivot_data = df.groupby(["model", "type"])[metric].mean().unstack()

        # 绘制柱状图
        pivot_data.plot(kind="bar", ax=ax, rot=0, width=0.8)

        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # 特殊处理延迟的单位
        if metric == "latency":
            ax.set_ylabel("Seconds", fontsize=10)
        else:
            ax.set_ylabel("Score", fontsize=10)

        # 添加数值标签
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=8
            )

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir = os.path.dirname(excel_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_path = os.path.join(output_dir, f"autotest_compare_plot_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"可视化图表已保存至：{plot_path}")