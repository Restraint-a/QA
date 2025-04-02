import os
import time
import torch
import logging
from Core.models import DocumentQASystem
from Core.document_loader import load_document
from Utils.utils import print_help, export_to_excel

def main():
    qa_system = DocumentQASystem()
    print_help()
    current_model = qa_system.current_model

    while True:
        user_input = input("\nYou: ").strip()

        # 指令处理
        if user_input.startswith("/"):
            if user_input.startswith("/switch"):
                model_name = user_input.split()[-1].lower()
                if model_name in qa_system.llm_registry:
                    qa_system.current_model = model_name
                    qa_system._release_model_resources()
                    qa_system.conversation_chain = None  # 强制重建对话链
                    print(f"🔄 已切换至 {model_name.upper()} 模型")
                    current_model = qa_system.current_model
                else:
                    print(f"⚠️ 可用模型：{list(qa_system.llm_registry.keys())}")
                continue

            elif user_input.startswith("/compare"):
                query = user_input[8:].strip()
                if not query:
                    print("❌ 请输入对比问题")
                    continue

                print(f"\n🔍 正在对比模型性能...")
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
                        print(f"❌ {name.upper()} 模型响应失败：{str(e)}")

                print("\n🆚 性能对比结果：")
                for model, data in results.items():
                    print(f"{model.upper()}:")
                    print(f"⏱️ 响应时间: {data['latency']}")
                    print(f"📝 响应示例: {data['response']}...\n")

                export_file = export_to_excel(results, query)
                if export_file:
                    print(f"\n📊 比较结果已导出至: {os.path.abspath(export_file)}")
                else:
                    print("\n⚠️ 导出失败，请检查日志")
                continue

            elif user_input == "/help":
                print_help()
                continue

            elif user_input == "/upload":
                file_path = input("📂 请输入本地文件路径: ").strip()
                if not os.path.exists(file_path):
                    print("❌ 文件不存在")
                    continue

                if load_document(qa_system, file_path):
                    print("✅ 文件已加载，现在可以提问！")
                continue

            else:
                print("❌ 未知指令，请输入 /help 查看可用指令")
                continue

        if user_input.lower() in ["exit", "quit"]:
            print("👋 退出对话")
            break

        # 如果输入中包含上传相关关键词则重复上传逻辑
        if any(word in user_input.lower() for word in ["上传", "文件", "文档", "upload"]):
            file_path = input("📂 请输入本地文件路径: ").strip()
            if not os.path.exists(file_path):
                print("❌ 文件不存在")
                continue

            if load_document(qa_system, file_path):
                print("✅ 文件已加载，现在可以提问！")
            continue

        # 构建或复用对话链
        if not qa_system.conversation_chain:
            from langchain.chains import ConversationChain  # 局部导入
            qa_system.conversation_chain = ConversationChain(
                llm=qa_system.llm_registry[qa_system.current_model],
                memory=qa_system.memory
            )

        # 处理问答请求
        try:
            if qa_system.qa_chain:
                result = qa_system.qa_chain({"query": user_input})
                response = f"{result['result']}\n\n📚 来源文档：{result['source_documents'][0].metadata['source']}"
            else:
                response = qa_system.conversation_chain.predict(input=user_input)

        except Exception as e:
            response = f"系统错误：{str(e)}"
            logging.error(f"处理请求失败: {str(e)}")

        print(f"{current_model}:", response)

if __name__ == "__main__":
    main()
