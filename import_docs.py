# import_docs.py
# 把这个文件放到 D:\FastAPIProject2\ 目录下
# 作用：把你准备好的Python技术文档快速导入知识库
# 
# 使用方法：
# 1. 把AI生成的Python技术文档保存成 .txt 文件，放到 D:\FastAPIProject2\docs\ 目录下
# 2. 运行：python import_docs.py

import os
from rag import add_documents

DOCS_DIR = "./docs"  # 把你的文档放在这个目录下


def import_all_docs():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"已创建 {DOCS_DIR} 目录，请把 .txt 文档放进去再运行")
        return

    txt_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"⚠️  {DOCS_DIR} 目录下没有 .txt 文件")
        print("   请把AI生成的Python技术文档保存为 .txt 文件放进去")
        return

    print(f"发现 {len(txt_files)} 个文档，开始导入...")

    for filename in txt_files:
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        topic = filename.replace(".txt", "")
        count = add_documents(
            texts=[content],
            metadatas=[{
                "source": filename,
                "category": "Python技术文档",
                "topic": topic,
            }]
        )
        print(f"  ✅ {filename} → 切分为 {count} 个chunk")

    print("\n导入完成！现在可以运行 eval_compare.py 了")


if __name__ == "__main__":
    import_all_docs()
