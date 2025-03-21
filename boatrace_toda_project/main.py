"""
ボートレース戸田予想ツールのメインスクリプト
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.ui.cli import BoatraceCliApp

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'main.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    メイン関数
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='ボートレース戸田予想ツール')
    parser.add_argument('--test', action='store_true', help='テストを実行する')
    parser.add_argument('--test-integration', action='store_true', help='統合テストを実行する')
    args = parser.parse_args()
    
    # 必要なディレクトリを作成
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
    
    # テストモード
    if args.test:
        logger.info("ユニットテストを実行します")
        sys.path.append(os.path.dirname(__file__))
        from tests.test_modules import run_tests
        run_tests()
        return
    
    # 統合テストモード
    if args.test_integration:
        logger.info("統合テストを実行します")
        sys.path.append(os.path.dirname(__file__))
        from tests.test_integration import run_integration_tests
        run_integration_tests()
        return
    
    # 通常モード
    logger.info("ボートレース戸田予想ツールを起動します")
    app = BoatraceCliApp()
    app.run()

if __name__ == "__main__":
    main()
