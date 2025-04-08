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

# Global variable for storing recent queries and responses
last_query = None
last_response = None
last_model = None

def process_command(command: str, qa_system: DocumentQASystem) -> bool:
    """Processing system commands"""
    global last_query, last_response, last_model
    
    command = command.strip().lower()

    if command.startswith("/switch"):
        model_name = command.split()[-1] if len(command.split()) > 1 else ""
        if model_name in qa_system.llm_registry:
            qa_system.current_model = model_name
            qa_system._release_model_resources()
            qa_system.conversation_chain = None
            print(f"Switched to {model_name.upper()} model")
            return True
        else:
            print(f"❌ Invalid model with available options:{list(qa_system.llm_registry.keys())}")
            return False

    # Handling user feedback - Agree
    elif command == "/like":
        if not last_query or not last_response:
            print("❌ No previous conversations found!")
            return False
        
        print("✅ Thank you for your feedback!")
        save_feedback_data(last_query, last_response, last_model, "like")
        return True
    
    # Handling User Feedback - Disagree
    elif command.startswith("/dislike"):
        if not last_query or not last_response:
            print("❌ No previous conversations found!")
            return False
        
        # Reasons for feedback
        reason = command[8:].strip() if len(command) > 8 else None
        if not reason:
            reason = input("Please briefly describe the reason for your dissatisfaction. ")
        
        print("🔄 Improved responses are being generated based on your feedback...")
        
        # Generate improved prompts
        improved_prompt = generate_improved_prompt(last_query, last_response, "dislike", reason)
        
        # Re-generate responses using improved prompts
        try:
            model = qa_system.llm_registry[qa_system.current_model]
            
            # Combining original queries and improving prompt words
            combined_prompt = f"{improved_prompt}\n\nOriginal question:{last_query}"
            
            # Re-generate the answer
            improved_response = model.invoke(combined_prompt)
            
            print(f"\n{qa_system.current_model.upper()} (After improvement):", improved_response)
            
            # Preservation of feedback and improved responses
            save_feedback_data(last_query, last_response, last_model, "dislike", 
                              improved_response, reason)
            
            # Update Recent Responses
            last_response = improved_response
            
        except Exception as e:
            error_msg = f"❌ Error when generating improved responses:{str(e)}"
            logging.error(error_msg)
            print(error_msg)
        
        return True

    # The /compare command processing section of main.py
    elif command.startswith("/compare"):
        query_part = command[8:].strip()
        if not query_part:
            print("Please enter a multi-line query (enter a blank line to end):")
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
            print("Please provide the query in the format: /compare [query]")
            return False

        print(f"\nThe responses of the models are being compared...")
        results = {}
        for name, model in qa_system.llm_registry.items():
            try:
                # Enforce garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Waiting for the system to stabilize
                time.sleep(2)
                
                # Record memory and GPU memory usage at the start
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
                
                # Record the GPU memory usage at the beginning of the record
                gpu_memory_before = qa_system.get_gpu_memory_usage()
                
                start_time = time.time()
                if qa_system.qa_chain:  # Check for document uploads
                    # Generating document-based answers with qa_chain
                    qa_system.qa_chain = RetrievalQA.from_chain_type(
                        llm=model,
                        chain_type="stuff",
                        retriever=qa_system.vector_db.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    result = qa_system.qa_chain({"query": query})
                    response = f"{result['result']}\n📚 Source:{result['source_documents'][0].metadata['source']}"
                else:
                    # No document upload, direct model call
                    response = model.invoke(query)
                end_time = time.time()
                
                # End-of-record memory usage
                memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
                memory_usage = memory_after - memory_before
                
                # Waiting for the system to stabilize
                time.sleep(2)
                
                # Enforce garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # GPU memory usage at the end
                gpu_memory_after = qa_system.get_gpu_memory_usage()
                
                # Calculating GPU Memory Changes
                gpu_memory_diff = {}
                if gpu_memory_before["available"] and gpu_memory_after["available"]:
                    for device_id, before_stats in gpu_memory_before["devices"].items():
                        after_stats = gpu_memory_after["devices"][device_id]

                        memory_diff = round(after_stats["used_memory_mb"] - before_stats["used_memory_mb"], 2)
                        
                        # Getting Peak Memory Usage
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
                
                # Calculate the speed of token consumption
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
                
                # Release of resources
                del response
                qa_system._release_model_resources()
            except Exception as e:
                print(f"{name.upper()} call went wrong:{str(e)}")

        print("\nCompare results:")
        for model_name, data in results.items():
            print(f"{model_name.upper()}:")
            print(f"Latency:{data['latency']}")
            print(f"Estimated token num:{data['tokens']}")
            print(f"Estimated token consumption rate:{data['tokens_per_second']} tokens/s")
            print(f"Response length: {data['response_length']} characters")
            print(f"Memory usage:{data['memory_usage']} MB")
            
            # Show GPU Memory Usage
            if "gpu_memory_diff" in data and data["gpu_memory_diff"]:
                print("GPU Memory Usage:")
                for device_id, gpu_stats in data["gpu_memory_diff"].items():
                    print(f"  {gpu_stats['device_name']}:")
                    print(f"    GPU Memory Usage Changes: {gpu_stats['memory_diff_mb']} MB")
                    if "peak_memory_mb" in gpu_stats:
                        print(f"    GPU Memory Usage: {gpu_stats['peak_memory_mb']} MB")
            else:
                print("GPU Memory Information: Unavailable")
            
            print(f"Response Preview:{data['response'][:100]}...\n")

        export_file = export_to_excel(results, query)
        if export_file:
            print(f"\nResults have been exported to:{os.path.abspath(export_file)}")
        else:
            print("❌ \nFailed to export results")
        return True

    # /autotest
    elif command.startswith("/autotest"):
        # Ask for the number of tests
        logic_count = int(input("The number of logic questions for testing:"))
        read_count = int(input("The number of reading questions for testing:"))
        math_count = int(input("The number of math questions for testing:"))

        # Call the relevant function to load the problem and test it
        print(f"Start automated testing with {logic_count} logic questions, {read_count} reading comprehension questions, {math_count} math questions...")
        results, standard_answers = perform_auto_test(qa_system, logic_count, read_count, math_count)

        # Export and visualize results
        export_file = Core.auto_test.export_to_excel(results, standard_answers)
        if export_file:
            print(f"\nTest results have been exported to:{os.path.abspath(export_file)}")
            visualize_results(export_file)  # Generate visualization charts
        else:
            print("❌ Failed to export results")

        return True

    elif command == "/help":
        print_help()
        return True

    elif command == "/upload":
        file_path = input("📂 Please enter the file path:").strip()
        if not os.path.exists(file_path):
            print("❌ File does not exist!")
            return False
        if load_document(qa_system, file_path):
            print("📄  Document Loaded Successfully")
            return True
        print("❌ Document Load Failure")
        return False

    else:
        print("❌ Unknown command, type /help for help")
        return False

def process_query(query: str, qa_system: DocumentQASystem, current_model: str) -> None:
    """处理用户查询"""
    global last_query, last_response, last_model
    
    try:
        if qa_system.qa_chain:
            result = qa_system.qa_chain({"query": query})
            response = f"{result['result']}\nSource:{result['source_documents'][0].metadata['source']}"
        else:
            if not qa_system.conversation_chain:
                from langchain.chains import ConversationChain
                qa_system.conversation_chain = ConversationChain(
                    llm=qa_system.llm_registry[qa_system.current_model],
                    memory=qa_system.memory
                )
            response = qa_system.conversation_chain.predict(input=query)

        print(f"\n{current_model.upper()}:", response)
        
        # Save recent queries and responses for feedback
        last_query = query
        last_response = response
        last_model = current_model
        
        # 提示用户可以提供反馈
        print("\n💬 You can use /like to agree, or /dislike [reason] to disagree.")

    except Exception as e:
        error_msg = f"❌ Handling errors:{str(e)}"
        logging.error(error_msg)
        print(error_msg)

def main():
    qa_system = DocumentQASystem()
    print_help()
    current_model = qa_system.current_model
    
    # Ensure that the feedback catalog exists
    os.makedirs("./feedback_data", exist_ok=True)

    while True:
        try:
            # Initializing Input Collection
            user_input = []
            print("\nYou: (Type in the content and enter a blank line to submit)")

            # Multi-line input loop
            while True:
                line = input().strip()

                # Exit command processing
                if line.lower() in ["exit", "quit","/exit","/quit"]:
                    print("👋  See you next time!")
                    return

                # The command is executed immediately
                if line.startswith("/"):
                    process_command(line, qa_system)
                    current_model = qa_system.current_model
                    break

                # A blank line indicates that the input is submitted
                if not line:
                    if user_input:
                        full_query = "\n".join(user_input)
                        print("🤖 Model Thinking......")
                        process_query(full_query, qa_system, current_model)
                    user_input = []
                    break

                user_input.append(line)

        except KeyboardInterrupt:
            print("\nEnter interrupt, enter exit to exit the program")
        except Exception as e:
            logging.error(f"System Error:{str(e)}")
            print("❌ An unexpected error has occurred, please try again")

if __name__ == "__main__":
    main()