# auto_test.py用于自动化测试，/autotest
# 根据指定数量题目类型，去问题集获取问题并输入到模型中，将模型返回的答案按照指定格式保存，并对比标准答案，计算答案质量和模型性能
import os
import re
import time
import logging
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from Core.models import DocumentQASystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def perform_auto_test(qa_system: DocumentQASystem, logic_count: int, read_count: int, math_count: int):
    """自动测试流程"""
    results = {}

    # 1. 加载逻辑题、阅读理解题、数学题并生成问题集合
    logic_questions = load_questions_from_directory('questions_set/logic', logic_count)
    read_questions = load_questions_from_directory('questions_set/read', read_count)
    math_questions = load_questions_from_directory('questions_set/math', math_count)

    # 载入标准答案
    standard_answers = {
        **load_standard_answers('questions_set/logic/logic_answer.txt'),
        **load_standard_answers('questions_set/read/read_answer.txt'),
        **load_standard_answers('questions_set/math/math_answer.txt')
    }

    # 模板提示词
    template = """
    You will be given a passage with several questions. Please answer each question strictly following the format below:

    1.1【Answer】B
    Explanation: The correct answer is B because...

    1.2【Answer】D
    Explanation: The correct answer is D because...

    Do not include any other explanations or remarks. Only provide the answers and explanations in the specified format.
    """

    # 2. 遍历所有模型，生成并记录答案
    for name, model in qa_system.llm_registry.items():
        print(f"测试模型 {name.upper()}...")

        # 对每种题型进行测试
        model_results = {"latency": [], "responses": [], "accuracy": [], "response_relevance": [],
                         "task_completion_rate": []}

        for question_set in [logic_questions, read_questions, math_questions]:
            for question in question_set:
                # 构造问题和提示词的结合体
                prompt = f"{template}\nPassage: {question['text']}\nQuestion: {question['text']}"

                start_time = time.time()
                # 处理问题并生成回答
                result = qa_system.qa_chain({"query": question['text']})
                parsed = parse_model_output_english(result)
                response = result['result']
                end_time = time.time()

                # 记录响应时间和其他指标
                latency = end_time - start_time
                model_results["latency"].append(latency)
                model_results["responses"].append(response)

                # 计算准确率、响应相关性、任务完成率
                completed = 0
                correct_answers = 0
                sim_scores = []

                for qid, model_ans in parsed.items():
                    std_ans = standard_answers.get(qid)
                    if not std_ans:
                        continue

                    # 计算选项匹配（准确率）
                    is_correct = model_ans["answer"] == std_ans["correct_option"]
                    if is_correct:
                        correct_answers += 1

                    # 计算解释的相似度（响应相关性）
                    sim_score = compute_similarity(std_ans["explanation"], model_ans["explanation"])
                    sim_scores.append(sim_score)

                    # 任务完成率（回答了多少道题）
                    completed += 1

                # 计算准确率、响应相关性、任务完成率
                accuracy = correct_answers / len(standard_answers)
                response_relevance = sum(sim_scores) / len(sim_scores) if sim_scores else 0
                task_completion_rate = completed / len(standard_answers)

                # 将评估结果保存
                model_results["accuracy"].append(accuracy)
                model_results["response_relevance"].append(response_relevance)
                model_results["task_completion_rate"].append(task_completion_rate)

        # 记录模型结果
        results[name] = model_results

    return results


# 英文格式解析函数
def parse_model_output_english(text):
    pattern = r'(\d+\.\d+)【Answer】([A-D])\s+Explanation:\s*(.*?)(?=\n\d+\.\d+【Answer】|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    return {
        qid: {'answer': answer, 'explanation': explanation.strip()}
        for qid, answer, explanation in matches
    }

# 相似度评分函数
def compute_similarity(text1, text2):
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


def load_questions_from_directory(directory_path, count):
    """从指定目录加载问题，返回指定数量的题目"""
    question_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    selected_files = question_files[:count]

    questions = []
    for file in selected_files:
        with open(os.path.join(directory_path, file), "r", encoding="utf-8") as f:
            text = f.read().strip()
            questions.append({"file": file, "text": text})  # 保存问题内容

    return questions


def load_standard_answers(answer_file_path):
    """从标准答案文件加载答案内容"""
    standard_answers = {}
    with open(answer_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_question = None
    for line in lines:
        line = line.strip()
        if line:
            # 假设每行格式为 “题号【答案】选项”
            match = re.match(r'(\d+\.\d+)【Answer】([A-D])\s+Explanation:\s*(.*?)(?=\n\d+\.\d+【Answer】|\Z)', line)
            if match:
                current_question = match.group(1)
                correct_option = match.group(2)
                standard_answers[current_question] = {"correct_option": correct_option}

    return standard_answers


def visualize_results(export_file):
    """生成可视化结果图"""
    df = pd.read_excel(export_file, sheet_name="Summary Metrics")

    # 画一个基本的条形图：展示各模型的响应时间、任务完成率、准确率等
    fig, ax = plt.subplots(figsize=(10, 6))
    df.set_index('Model').plot(kind='bar', ax=ax)
    ax.set_ylabel('Metrics')
    ax.set_title('Model Performance Comparison')

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

def export_to_excel(results, query, standard_answers, output_dir="."):
    try:
        safe_query = query.replace("\n", " ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_detailed_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)

        # 保存每道题的数据
        export_data = []
        summary_data = []

        for model_name, result in results.items():
            parsed = parse_model_output_english(result["response"])
            latency = result.get("latency", None)

            total_questions = len(standard_answers)
            completed = 0
            sim_scores = []
            correct_answers = 0

            for qid, model_ans in parsed.items():
                std_ans = standard_answers.get(qid)
                if not std_ans:
                    continue

                # 计算选项匹配和相似度
                is_correct = model_ans["answer"] == std_ans["correct_option"]
                sim_score = compute_similarity(std_ans["explanation"], model_ans["explanation"])
                eval_remark = (
                    "Correct answer, explanation matches well" if is_correct and sim_score >= 0.7 else
                    "Correct answer, explanation needs improvement" if is_correct else
                    "Wrong answer, explanation is relevant" if sim_score >= 0.7 else
                    "Wrong answer, poor explanation"
                )

                export_data.append({
                    "Model": model_name.upper(),
                    "Query": safe_query,
                    "Question ID": qid,
                    "Latency": latency,
                    "Correct Answer": std_ans["correct_option"],
                    "Model Answer": model_ans["answer"],
                    "Match": "✅" if is_correct else "❌",
                    "Similarity": round(sim_score, 2),
                    "Evaluation": eval_remark,
                    "Model Explanation": model_ans["explanation"]
                })

                completed += 1
                if is_correct:
                    correct_answers += 1
                sim_scores.append(sim_score)

            task_completion_rate = completed / total_questions
            avg_similarity = sum(sim_scores) / len(sim_scores) if sim_scores else 0
            accuracy = correct_answers / total_questions

            summary_data.append({
                "Model": model_name.upper(),
                "Latency": latency,
                "Task Completion Rate": round(task_completion_rate, 2),
                "Avg Response Relevance (TF-IDF)": round(avg_similarity, 2),
                "Accuracy": round(accuracy, 2)
            })

        df_per_question = pd.DataFrame(export_data)
        df_summary = pd.DataFrame(summary_data)

        # 保存两个 sheet
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df_per_question.to_excel(writer, sheet_name="Per Question Analysis", index=False)
            df_summary.to_excel(writer, sheet_name="Summary Metrics", index=False)

        return filepath

    except Exception as e:
        logging.error(f"导出失败: {str(e)}")
        return None