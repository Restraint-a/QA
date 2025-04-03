# main.py
import os
import time
import torch
import logging
from Core.models import DocumentQASystem
from Core.document_loader import load_document
from Utils.utils import print_help, export_to_excel

def process_command(command: str, qa_system: DocumentQASystem) -> bool:
    """处理系统命令"""
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
            print(f"无效模型，可用选项：{list(qa_system.llm_registry.keys())}")
            return False

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
                start_time = time.time()
                response = model.invoke(query)
                end_time = time.time()
                latency = end_time - start_time
                results[name] = {
                    "response": response,
                    "latency": f"{latency:.2f}s"
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
            print(f"响应预览：{data['response'][:200]}...\n")

        export_file = export_to_excel(results, query)
        if export_file:
            print(f"\n已导出结果至：{os.path.abspath(export_file)}")
        else:
            print("\n导出结果失败")
        return True

    elif command == "/help":
        print_help()
        return True

    elif command == "/upload":
        file_path = input("请输入文件路径：").strip()
        if not os.path.exists(file_path):
            print("文件不存在")
            return False
        if load_document(qa_system, file_path):
            print("文档加载成功")
            return True
        print("文档加载失败")
        return False

    else:
        print("未知命令，输入/help查看帮助")
        return False

def process_query(query: str, qa_system: DocumentQASystem, current_model: str) -> None:
    """处理用户查询"""
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

    except Exception as e:
        error_msg = f"处理错误：{str(e)}"
        logging.error(error_msg)
        print(error_msg)

def main():
    qa_system = DocumentQASystem()
    print_help()
    current_model = qa_system.current_model

    while True:
        try:
            # 初始化输入收集
            user_input = []
            print("\nYou: (输入内容，连按两次回车提交)")

            # 多行输入循环
            while True:
                line = input().strip()

                # 退出指令处理
                if line.lower() in ["exit", "quit"]:
                    print("再见！")
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
                        print("Model思考中......")
                        process_query(full_query, qa_system, current_model)
                    user_input = []
                    break

                user_input.append(line)

        except KeyboardInterrupt:
            print("\n输入中断，输入 exit 退出程序")
        except Exception as e:
            logging.error(f"系统错误：{str(e)}")
            print("发生意外错误，请重新尝试")

if __name__ == "__main__":
    main()