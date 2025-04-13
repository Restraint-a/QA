# utils.py
import os
import logging
import pandas as pd
from datetime import datetime


DEFAULT_PATH = "./compare_output"
FEEDBACK_PATH = "./feedback_data"

def print_help():
    print("""
🔧 System Commands:
    /switch [mistral|qwen|gemma3|]  - Switch LLM
    /compare [Question]         - Compare model performance
    /help                  - Show help
    /upload                - Upload a document to enter document Q&A mode
    /reset                 - Switch back to normal conversation mode
    /autotest              - Automatic testing
    /like                  - Agree with previous answer
    /dislike [Reason]         - Disagree with previous answer, and provide reasons
    /exit/quit              - Exit system
""")

def export_to_excel(results, query):
    try:
        safe_query = query.replace("\n", " ")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.xlsx"

        # Assembly data
        data = []
        for model_name, result in results.items():
            # Basic Information
            row = {
                "Model": model_name.upper(),
                "Query": safe_query,
                "Latency": result["latency"],
                "Tokens": result.get("tokens", "N/A"),
                "Tokens/Second": result.get("tokens_per_second", "N/A"),
                "Response Length": result.get("response_length", "N/A"),
                "Memory Usage (MB)": result.get("memory_usage", "N/A"),
            }
            
            # Add GPU Memory Information
            if "gpu_memory_diff" in result and result["gpu_memory_diff"]:
                for device_id, gpu_stats in result["gpu_memory_diff"].items():
                    # Only memory usage variations and peaks are preserved
                    row[f"Memory_Diff_MB"] = gpu_stats["memory_diff_mb"]
                    if "peak_memory_mb" in gpu_stats:
                        row[f"Peak_Memory_MB"] = gpu_stats["peak_memory_mb"]
                
            
            # Add response content
            row["Response"] = result["response"]
            data.append(row)
            
        # Create DataFrame and export
        df = pd.DataFrame(data)
        
        # Make sure the output directory exists
        os.makedirs(DEFAULT_PATH, exist_ok=True)
        output_path = os.path.join(DEFAULT_PATH, filename)
        
        # Export to Excel
        df.to_excel(output_path, index=False)
        return output_path
    except Exception as e:
        logging.error(f"Export failed: {str(e)}")
        return None

def generate_improved_prompt(query, response, feedback, reason=None):
    """Generate improved prompt words based on user feedback"""
    if feedback == "like":
        # If the user likes the response, record this pattern
        return None  # No need to change the cue word
    
    # Generate improved prompt words when users don't like the answer
    improved_prompt = f"""
        My previous answer was not good enough.The original question was:
        "{query}"

        My answer was:
        "{response}"

        User feedback: Dissatisfied
        """
    
    if reason:
        improved_prompt += f"Reason: {reason}\n"
    
    improved_prompt += """
        Please provide a better answer based on the above feedback.Notes.
        1. keep your answer accurate and relevant
        2. Provide more detailed explanations and examples
        3. Make sure the answer is logical and easy to understand
        """
    
    return improved_prompt

def save_feedback_data(query, response, model_name, feedback, improved_response=None, reason=None):
    """Save user feedback data"""
    try:
        os.makedirs(FEEDBACK_PATH, exist_ok=True)
        
        # Prepare feedback data
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
        
        # Create or append to a feedback file
        feedback_file = os.path.join(FEEDBACK_PATH, "user_feedback.xlsx")
        
        # Append data if file exists
        if os.path.exists(feedback_file):
            existing_df = pd.read_excel(feedback_file)
            new_df = pd.DataFrame(feedback_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(feedback_file, index=False)
        else:
            # Create a new file
            pd.DataFrame(feedback_data).to_excel(feedback_file, index=False)
            
        return feedback_file
    except Exception as e:
        logging.error(f"Failed to save feedback: {str(e)}")
        return None