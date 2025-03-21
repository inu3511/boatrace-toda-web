import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import argparse

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.boatrace_data_collector import BoatraceDataCollector
from data_collection.boatrace_data_analyzer import BoatraceDataAnalyzer

def main():
    """
    ボートレース戸田のデータ収集と分析を実行するメイン関数
    """
    parser = argparse.ArgumentParser(description='ボートレース戸田のデータ収集・分析ツール')
    parser.add_argument('--collect', action='store_true', help='データ収集を実行する')
    parser.add_argument('--analyze', action='store_true', help='データ分析を実行する')
    parser.add_argument('--start-date', type=str, help='データ収集開始日（YYYYMMDD形式）')
    parser.add_argument('--end-date', type=str, help='データ収集終了日（YYYYMMDD形式）')
    parser.add_argument('--data-file', type=str, help='分析対象のデータファイル')
    
    args = parser.parse_args()
    
    # デフォルトの日付範囲（過去30日間）
    if args.start_date is None or args.end_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        args.end_date = end_date.strftime('%Y%m%d')
        args.start_date = start_date.strftime('%Y%m%d')
    
    # データ収集
    if args.collect:
        print(f"データ収集を開始します（期間: {args.start_date} から {args.end_date}）")
        collector = BoatraceDataCollector()
        results = collector.collect_data_for_period(args.start_date, args.end_date)
        
        if results is not None:
            print(f"データ収集が完了しました。{len(results)}件のレース結果を取得しました。")
            
            # デフォルトのデータファイル名を設定
            if args.data_file is None:
                args.data_file = os.path.join(collector.data_dir, f"toda_race_results_{args.start_date}_to_{args.end_date}.csv")
        else:
            print("データ収集に失敗しました。")
            return
    
    # データ分析
    if args.analyze:
        if args.data_file is None:
            print("分析対象のデータファイルが指定されていません。")
            return
        
        print(f"データ分析を開始します（ファイル: {args.data_file}）")
        analyzer = BoatraceDataAnalyzer()
        df = analyzer.load_data(args.data_file)
        
        if df is not None:
            # コース別統計情報の分析
            stats = analyzer.analyze_course_statistics(df)
            analyzer.plot_course_statistics(stats)
            
            # 季節別トレンドの分析
            seasonal_stats = analyzer.analyze_seasonal_trends(df)
            analyzer.plot_seasonal_trends(seasonal_stats)
            
            print("データ分析が完了しました。")
        else:
            print("データ分析に失敗しました。")
            return

if __name__ == "__main__":
    main()
