# analyze_chunking.py
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from Utils.chunking_analyzer import ChunkingAnalyzer, run_analysis

def visualize_from_excel(excel_path, output_dir=None):
    """Generate Visual Charts from Existing Excel Results Files

    Args.
        excel_path: path to the Excel result file
        output_dir: output directory, defaults to the directory where the Excel file is located.

    Returns.
        The path of the generated chart, if it fails, return None.
    """
    try:
        # Check if the file exists
        if not os.path.isfile(excel_path):
            print(f"❌ Excel file does not exist. {excel_path}")
            return None
            
        # Setting the output directory
        if output_dir is None:
            output_dir = os.path.dirname(excel_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Reading Excel files
        print(f"📊 Load data from Excel file. {excel_path}")
        df = pd.read_excel(excel_path)
        
        if df.empty:
            print("❌ No data in Excel file")
            return None
            
        print(f"✅ Successfully loaded data: {len(df)} rows x {len(df.columns)} columns")
        print(f"📋 Column names: {list(df.columns)}")
        
        # Creating Charts in English
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = True
        
        # Creating Charts
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Relationship between chunk size and processing time
        axs[0, 0].set_title('Chunk Size vs Processing Time')
        for overlap in df['chunk_overlap'].unique():
            subset = df[df['chunk_overlap'] == overlap]
            line, = axs[0, 0].plot(subset['chunk_size'], subset['total_time'], 'o-', label=f'Overlap={overlap}')
            
            # Adding data point labels
            for x, y in zip(subset['chunk_size'], subset['total_time']):
                axs[0, 0].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0, 5), ha='center')
                
        axs[0, 0].set_xlabel('Chunk Size')
        axs[0, 0].set_ylabel('Total Time (s)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # 2. Relationship between chunk size and memory usage
        axs[0, 1].set_title('Chunk Size vs Memory Usage')
        for overlap in df['chunk_overlap'].unique():
            subset = df[df['chunk_overlap'] == overlap]
            line, = axs[0, 1].plot(subset['chunk_size'], subset['ram_usage_mb'], 'o-', label=f'Overlap={overlap}')
            
            # Adding data point labels
            for x, y in zip(subset['chunk_size'], subset['ram_usage_mb']):
                axs[0, 1].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0, 5), ha='center')
                
        axs[0, 1].set_xlabel('Chunk Size')
        axs[0, 1].set_ylabel('RAM Usage (MB)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # 3. Relationship between number of chunks and processing time
        axs[1, 0].set_title('Number of Chunks vs Processing Time')
        
        # Plotting curves grouped by overlap size
        for overlap in df['chunk_overlap'].unique():
            subset = df[df['chunk_overlap'] == overlap]
            # Sort by number of chunks
            subset = subset.sort_values('num_chunks')
            line, = axs[1, 0].plot(subset['num_chunks'], subset['total_time'], 'o-', 
                                  label=f'Overlap={overlap}')
            
            # Adding data point labels
            for x, y in zip(subset['num_chunks'], subset['total_time']):
                axs[1, 0].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0, 5), ha='center')
                
        axs[1, 0].set_xlabel('Number of Chunks')
        axs[1, 0].set_ylabel('Total Time (s)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # 4. GPU memory usage (if any)
        axs[1, 1].set_title('Chunk Size vs GPU Memory Usage')
        gpu_cols = [col for col in df.columns if 'gpu_' in col and 'diff_mb' in col]
        
        if gpu_cols:
            for gpu_col in gpu_cols:
                device_id = gpu_col.split('_')[1]
                for overlap in df['chunk_overlap'].unique():
                    subset = df[df['chunk_overlap'] == overlap]
                    line, = axs[1, 1].plot(subset['chunk_size'], subset[gpu_col], 'o-', 
                                         label=f'GPU {device_id}, Overlap={overlap}')
                    
                    # Add data point labels
                    for x, y in zip(subset['chunk_size'], subset[gpu_col]):
                        axs[1, 1].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                         xytext=(0, 5), ha='center')
                        
            axs[1, 1].set_xlabel('Chunk Size')
            axs[1, 1].set_ylabel('GPU Memory Change (MB)')
            axs[1, 1].legend()
            axs[1, 1].grid(True)
        else:
            axs[1, 1].text(0.5, 0.5, 'GPU Data Not Available', ha='center', va='center', fontsize=14)
            axs[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save Chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"chunking_analysis_viz_{timestamp}.png")
        plt.savefig(output_path)
        print(f"✅ Charts have been saved to: {output_path}")
        
        # Show Chart
        plt.show()
        
        return output_path
        
    except Exception as e:
        print(f"❌ Errors during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Command Line Entry to the Chunking Policy Performance Analysis Tool"""
    parser = argparse.ArgumentParser(description="Analyzing the impact of different chunking strategies on memory and graphics performance")
    
    parser.add_argument(
        "--file", "-f", 
        type=str, 
        required=False,
        help="Path to the document to be analyzed"
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default="mistral",
        help="Ollama model name, default is mistral"
    )
    
    parser.add_argument(
        "--chunk-sizes", 
        type=int, 
        nargs="+", 
        default=[500, 1000, 1500, 2000],
        help="List of chunk sizes to test, for example:--chunk-sizes 500 1000 1500 2000"
    )
    
    parser.add_argument(
        "--chunk-overlaps", 
        type=int, 
        nargs="+", 
        default=[50, 100, 200],
        help="List of chunked overlap sizes to test, for example:--chunk-overlaps 50 100 200"
    )
    
    parser.add_argument(
        "--query", "-q", 
        type=str, 
        default="Please summarize the main points of this document",
        help="Query for testing"
    )
    
    parser.add_argument(
        "--visualize-excel", "-v", 
        type=str,
        help="Generate visual charts from existing Excel results files"
    )
    
    args = parser.parse_args()
    
    # If an Excel file is specified, the visualization chart is generated directly
    if args.visualize_excel:
        print("="*50)
        print("📊 Generate visual charts from Excel files")
        print("="*50)
        image_path = visualize_from_excel(args.visualize_excel)
        if image_path:
            print(f"✅ Visualization charts have been generated: {image_path}")
        else:
            print("❌ Fail to generate visualization charts")
        return 0 if image_path else 1
    
    # 否则执行完整的分析流程
    if not args.file:
        print("Error: File path not specified.Please specify the document to be analyzed using the --file parameter, or the Excel results file using the --visualize-excel parameter.")
        return 1
        
    # 验证文件路径
    if not os.path.isfile(args.file):
        print(f"Error: File does not exist - {args.file}")
        return 1
    
    try:
        print("="*50)
        print("📊 Chunking Strategy Performance Analysis Tool")
        print("="*50)
        print(f"📄 Analyzing document: {args.file}")
        print(f"🤖 Using model: {args.model}")
        print(f"📏 Chunk size to test: {args.chunk_sizes}")
        print(f"🔄 Overlap size to test: {args.chunk_overlaps}")
        print(f"❓ Test query: {args.query}")
        print("="*50)
        
        # Creating an Analyzer
        analyzer = ChunkingAnalyzer(model_name=args.model)
        
        # Run analyzer
        results = analyzer.analyze_chunking_strategy(
            file_path=args.file,
            chunk_sizes=args.chunk_sizes,
            chunk_overlaps=args.chunk_overlaps,
            query=args.query
        )
        
        # Export results
        excel_path = analyzer.export_results()
        
        # Visualization results
        image_path = analyzer.visualize_results()
        
        # If the built-in visualization fails, try using a new function
        if not image_path and excel_path:
            print("\n⚠️ Built-in visualization fails, try using direct visualization function...")
            image_path = visualize_from_excel(excel_path)
        
        print("\n" + "="*50)
        print("✅ Analysis complete!")
        if excel_path:
            print(f"📊 Excel results: {excel_path}")
        if image_path:
            print(f"📈 Visualization results: {image_path}")
        print("="*50)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())