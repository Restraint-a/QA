import os
import logging
import warnings

# 屏蔽不必要的警告
warnings.filterwarnings("ignore")

# 日志配置
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 环境变量设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
