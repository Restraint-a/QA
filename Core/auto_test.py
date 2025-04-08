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
    Load questions and standard answers
    :param question_type: question type (logic/math)
    :param question_set_dir: Question bank root directory
    :return: structured question data {qid: {"question": ..., "answer": ...}}
    """
    data = {}
    base_path = os.path.join(question_set_dir, question_type)

    if question_type == "math":
        # Load Math Problems
        with open(os.path.join(base_path, "math.txt"), "r", encoding="utf-8") as f:
            content = f.read()
        questions = re.split(r"\n\d+\.", content)  # Split by number
        questions = [q.strip() for q in questions if q.strip()]

        # Loading math answers
        with open(os.path.join(base_path, "math_answer.txt"), "r", encoding="utf-8") as f:
            answers = re.findall(r"(\d+)【Answer】(\w+)", f.read())
            answer_map = {qid: option.upper() for qid, option in answers}

        # Combine data
        for idx, q in enumerate(questions, 1):
            data[f"math_{idx}"] = {
                "question": q,
                "answer_option": answer_map.get(str(idx), ""),
                "answer_analysis": ""  # Math questions have no analysis text
            }

    elif question_type in ["logic", "read"]:
        for filename in os.listdir(base_path):
            if filename.endswith(".txt") and "answer" not in filename:
                # Extracts the question number from the file name, e.g. "logic_3.txt" extracts "3".
                file_parts = filename.split(".")
                if len(file_parts) > 0:
                    file_base = file_parts[0]  # For example, "logic_3"
                    qid_parts = file_base.split("_")
                    if len(qid_parts) > 1:
                        qid = qid_parts[1]  # Extract the question number, e.g. "3"

                        # Read the question file
                        with open(os.path.join(base_path, filename), "r", encoding="utf-8") as f:
                            content = f.read()

                            # Extract the last question (usually the last part of the question)
                            question_parts = content.split("\n\n")
                            for i in range(len(question_parts)-1, -1, -1):
                                if question_parts[i].strip():
                                    question = question_parts[i].strip()
                                    break
                            else:
                                question = content.strip()

                        # Load answer file
                        answer_file = os.path.join(base_path, f"{file_base}_answer.txt")
                        if os.path.exists(answer_file):
                            with open(answer_file, "r", encoding="utf-8") as f:
                                answer_content = f.read().strip()

                                # Match question number and answer, supporting multiple formats
                                # e.g., "3.【Answer】B" or "3【Answer】B"
                                answer_match = re.search(r"(\d+)\.?【Answer】(\w+)([\s\S]*)", answer_content)
                                #print(answer_match)
                                if answer_match:
                                    answer_num = answer_match.group(1)
                                    option = answer_match.group(2).upper() if answer_match.group(2) else ""
                                    analysis = answer_match.group(3).strip() if answer_match.group(3) else ""

                                    # Ensure question number matches
                                    if answer_num == qid:
                                        data[f"{question_type}_{qid}"] = {
                                            "question": question,
                                            "answer_option": option,
                                            "answer_analysis": analysis
                                        }
                                        print(f"Successfully loaded {question_type}_{qid} question")
                                    else:
                                        print(f"Warning: Question number {answer_num} in {answer_file} does not match question number {qid} in filename")
                                else:
                                    print(f"Warning: Unable to extract answer from {answer_file}")
                        else:
                            print(f"Warning: Answer file {answer_file} does not exist")

    return data

def get_model_response(qa_system: DocumentQASystem, question: str) -> Tuple[str, float]:
    """
    Get model response and calculate latency
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
    Calculate accuracy and relevance
    """
    metrics = {
        "accuracy": 0.0,
        "relevance": 0.0,
        "completed_rate": 0.0
    }

    # Extract model answer option
    option_match = re.search(r"\b([A-D])\b", model_answer)
    model_option = option_match.group(1).upper() if option_match else ""
    print(f"Processing question: {item.get('question', '')[:30]}... Model option: {model_option}")

    # Accuracy evaluation (only for logic and math questions)
    if q_type in ["logic", "math"]:
        correct_option = item.get("answer_option", "").upper()
        metrics["accuracy"] = 1.0 if model_option == correct_option else 0.0

    # Relevance evaluation (for logic and reading questions)
    if q_type in ["logic", "read"]:
        metrics["relevance"] = compute_similarity(model_answer, item["answer_analysis"])

    # Completion rate evaluation (for logic questions only)
    if q_type == "logic":
        metrics["completed_rate"] = min(max(metrics["relevance"] * 1.5, 0.0), 1.0)  # Higher relevance means higher completion rate

    return metrics

# Similarity scoring function
# Scheme 1: TF-IDF based cosine similarity (lightweight)
def compute_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity (range 0~1)
    """
    # Preprocess text (optional)
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()

    # Empty text handling
    if not text1 or not text2:
        return 0.0

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(max(similarity, 0.0), 2)  # Ensure non-negative


# ---------- OR ---------- #

# Option 2: Sentence-BERT based semantic similarity (high precision, requires GPU acceleration)
# from sentence_transformers import SentenceTransformer
#
# # Load pre-trained model (first run requires download)
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#
#
# def compute_similarity(text1: str, text2: str) -> float:
#     """
#     Calculate semantic similarity (range 0~1)
#     """
#     # Encode text
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
    Execute automatic testing and return results
    """
    results = []
    standard_answers = {}

    # Load questions
    logic_data = load_questions_and_answers("logic")
    read_data = load_questions_and_answers("read")
    math_data = load_questions_and_answers("math")
    
    # Print the number of loaded questions
    print(f"Number of logic questions loaded: {len(logic_data)}")
    print(f"Number of reading questions loaded: {len(read_data)}")
    print(f"Number of math questions loaded: {len(math_data)}")
    
    # If the requested number of questions exceeds the available questions, adjust to the maximum available
    logic_count = min(logic_count, len(logic_data))
    read_count = min(read_count, len(read_data))
    math_count = min(math_count, len(math_data))
    
    print(f"Actual number of logic questions tested: {logic_count}")
    print(f"Actual number of reading questions tested: {read_count}")
    print(f"Actual number of math questions tested: {math_count}")

    # Iterate through all registered models
    for model_name in qa_system.llm_registry.keys():
        qa_system.current_model = model_name
        qa_system._release_model_resources()  # Release resources from previous model

        # Test logic questions
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

        # Test reading comprehension questions
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

        # Test math questions
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
    Export results to Excel (save to /compare_output directory)
    """
    # Ensure directory exists
    output_dir = "./compare_output"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    filename = os.path.join(output_dir, f"autotest_{time.strftime('%Y%m%d-%H%M%S')}.xlsx")
    df.to_excel(filename, index=False)
    return filename


def visualize_results(excel_path: str):
    """Generate visualization report"""
    df = pd.read_excel(excel_path)
    # Set font support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # Priority list of fonts
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
    # Create canvas and subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle("Multi-models comparisons", fontsize=16, y=0.95)

    # ================== Accuracy comparison ==================
    ax1 = axes[0, 0]
    accuracy_data = df[df['type'].isin(['logic', 'math'])] \
                        .groupby(['model', 'type'])['accuracy'].mean() * 100
    if not accuracy_data.empty:
        accuracy_data.unstack().plot.bar(ax=ax1)
        ax1.set_title('Accuracy Comparison', pad=15)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 110)
        ax1.tick_params(axis='x', rotation=45)
        # Adding numeric labels
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.1f%%', padding=3)
    else:
        ax1.text(0.5, 0.5, 'No Accuracy data', ha='center', va='center')

    # ================== Relevance comparison ==================
    ax2 = axes[0, 1]
    relevance_data = df[df['type'].isin(['logic', 'read'])] \
                         .groupby(['model', 'type'])['relevance'].mean() * 100
    if not relevance_data.empty:
        relevance_data.unstack().plot.bar(ax=ax2, color=['#1f77b4', '#ff7f0e'])
        ax2.set_title('Relevance Comparison', pad=15)
        ax2.set_ylabel('Relevance (%)')
        ax2.set_ylim(0, 110)
        ax2.tick_params(axis='x', rotation=45)
        # Adding numeric labels
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f%%', padding=3)
    else:
        ax2.text(0.5, 0.5, 'No Relevance data', ha='center', va='center')

    # ================== Completion comparison ==================
    ax3 = axes[1, 0]
    completion_data = df[df['type'] == 'logic'] \
                          .groupby('model')['completed_rate'].mean() * 100
    if not completion_data.empty:
        bars = completion_data.plot.bar(ax=ax3, color='#2ca02c')
        ax3.set_title('Logic Completion Comparison', pad=15)
        ax3.set_ylabel('completion (%)')
        ax3.set_ylim(0, 110)
        ax3.tick_params(axis='x', rotation=45)
        # Adding numeric labels
        ax3.bar_label(bars.containers[0], fmt='%.1f%%', padding=3)
    else:
        ax3.text(0.5, 0.5, 'No Completion data', ha='center', va='center')

    # ================== Latency comparison ==================
    ax4 = axes[1, 1]
    latency_data = df.groupby(['model', 'type'])['latency'].mean().unstack()
    if not latency_data.empty:
        latency_data.plot.bar(ax=ax4)
        ax4.set_title('Average Latency Comparison', pad=15)
        ax4.set_ylabel('Latency (s)')
        ax4.tick_params(axis='x', rotation=45)
        for container in ax4.containers:
            ax4.bar_label(container, fmt='%.1f', padding=3)
    else:
        ax4.text(0.5, 0.5, 'No Latency data', ha='center', va='center')

    # Save chart
    plot_path = excel_path.replace(".xlsx", "_visual.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Visualization chart saved to: {plot_path}")