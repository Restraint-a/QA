# 分块策略性能分析工具

## 简介

这个工具用于分析不同文本分块策略对内存和显存性能的影响。在大型语言模型处理长文本时，分块策略对性能和资源使用有显著影响。本工具可以帮助您测试不同的分块大小和重叠设置，找到最佳的性能与资源使用平衡点。

## 功能特点

- 分析不同分块大小对处理时间的影响
- 测量不同分块策略下的内存(RAM)使用情况
- 监控GPU显存使用变化和峰值
- 评估分块数量与处理效率的关系
- 生成详细的Excel报告和可视化图表
- 与现有Ollama模型无缝集成
- 完全独立运行，不影响main.py的功能

## 安装要求

本工具依赖于以下Python库：

```
pandas
matplotlib
numpy
torch
psutil
langchain
langchain_community
```

这些依赖项已包含在项目的`requirements.txt`中。

## 使用方法

### 命令行使用

```bash
python analyze_chunking.py --file <文档路径> [选项]
```

### 参数说明

- `--file`, `-f`: 要分析的文档路径（必需）
- `--model`, `-m`: Ollama模型名称，默认为"mistral"
- `--chunk-sizes`: 要测试的分块大小列表，例如：`--chunk-sizes 500 1000 1500 2000`
- `--chunk-overlaps`: 要测试的分块重叠大小列表，例如：`--chunk-overlaps 50 100 200`
- `--query`, `-q`: 用于测试的查询语句，默认为"请总结这篇文档的主要内容"

### 示例

```bash
# 基本用法
python analyze_chunking.py --file "d:\NLP\QA\test.txt"

# 指定模型和分块参数
python analyze_chunking.py --file "d:\NLP\QA\test.txt" --model "qwen" --chunk-sizes 300 600 900 --chunk-overlaps 30 60 90

# 自定义查询
python analyze_chunking.py --file "d:\NLP\QA\test.txt" --query "这篇文章的主要人物是谁？"
```

### 在代码中使用

您也可以在自己的Python代码中导入并使用分析工具：

```python
from Utils.chunking_analyzer import ChunkingAnalyzer

# 创建分析器实例
analyzer = ChunkingAnalyzer(model_name="mistral")

# 运行分析
results = analyzer.analyze_chunking_strategy(
    file_path="path/to/document.txt",
    chunk_sizes=[500, 1000, 1500],
    chunk_overlaps=[50, 100, 200]
)

# 导出结果
analyzer.export_results()

# 可视化结果
analyzer.visualize_results()
```

## 输出说明

分析完成后，工具会生成两种输出：

1. **Excel报告**：包含所有测试配置的详细性能数据，保存在`chunking_analysis_results`目录下
2. **可视化图表**：四个图表展示不同分块策略的性能对比，包括：
   - 块大小与处理时间的关系
   - 块大小与内存使用的关系
   - 块数量与处理时间的关系
   - 块大小与GPU内存使用的关系

## 性能指标解释

- **处理时间**：完成文档分块、向量化和查询的总时间（秒）
- **RAM使用**：处理过程中的内存增长（MB）
- **GPU内存差异**：处理前后的GPU显存变化（MB）
- **GPU内存峰值**：处理过程中的最大GPU显存使用量（MB）
- **块数量**：文档被分割成的文本块数量

## 优化建议

根据分析结果，您可以考虑以下优化策略：

1. **平衡块大小**：较大的块可能减少总块数，但可能增加单块处理的内存需求
2. **调整重叠度**：较大的重叠可能提高检索质量，但会增加总块数和处理时间
3. **考虑硬件限制**：在显存有限的环境中，可能需要选择较小的块大小
4. **任务特性**：对于需要更多上下文的复杂查询，可能需要增加块大小和重叠

## 注意事项

- 分析过程可能需要较长时间，特别是对于大型文档或多个参数组合
- 确保您的系统有足够的内存和GPU资源来运行分析
- 分析结果可能因文档内容、模型和硬件环境而异