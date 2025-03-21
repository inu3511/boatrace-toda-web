import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime, timedelta
import json

class BoatraceDataAnalyzer:
    """
    ボートレース戸田のデータを分析するクラス
    """
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = os.path.join(os.getcwd(), "data")
        else:
            self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 結果の保存先ディレクトリ
        self.results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # グラフの保存先ディレクトリ
        self.graphs_dir = os.path.join(self.results_dir, "graphs")
        os.makedirs(self.graphs_dir, exist_ok=True)
    
    def load_data(self, file_path):
        """
        CSVファイルからデータを読み込む
        
        Parameters:
        -----------
        file_path : str
            CSVファイルのパス
            
        Returns:
        --------
        pandas.DataFrame
            読み込んだデータのデータフレーム
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None
    
    def analyze_course_statistics(self, df):
        """
        コース別の統計情報を分析する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            レース結果のデータフレーム
            
        Returns:
        --------
        dict
            コース別統計情報
        """
        if df is None or df.empty:
            return None
        
        # 1着のコース別集計
        first_place_counts = df['first'].value_counts().sort_index()
        first_place_percentage = (first_place_counts / len(df) * 100).round(1)
        
        # 2着のコース別集計
        second_place_counts = df['second'].value_counts().sort_index()
        second_place_percentage = (second_place_counts / len(df) * 100).round(1)
        
        # 3着のコース別集計
        third_place_counts = df['third'].value_counts().sort_index()
        third_place_percentage = (third_place_counts / len(df) * 100).round(1)
        
        # 3連単の組み合わせ集計
        trifecta_combinations = df.apply(lambda row: f"{row['first']}-{row['second']}-{row['third']}", axis=1)
        top_combinations = trifecta_combinations.value_counts().head(10)
        
        # 結果を辞書にまとめる
        stats = {
            'first_place': {
                'counts': first_place_counts.to_dict(),
                'percentage': first_place_percentage.to_dict()
            },
            'second_place': {
                'counts': second_place_counts.to_dict(),
                'percentage': second_place_percentage.to_dict()
            },
            'third_place': {
                'counts': third_place_counts.to_dict(),
                'percentage': third_place_percentage.to_dict()
            },
            'top_combinations': top_combinations.to_dict()
        }
        
        # 結果をJSONファイルに保存
        output_file = os.path.join(self.results_dir, "course_statistics.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"Course statistics saved to {output_file}")
        
        return stats
    
    def plot_course_statistics(self, stats):
        """
        コース別統計情報をグラフ化する
        
        Parameters:
        -----------
        stats : dict
            コース別統計情報
            
        Returns:
        --------
        None
        """
        if stats is None:
            return
        
        # プロットのスタイル設定
        plt.style.use('ggplot')
        sns.set(font='IPAexGothic')  # 日本語フォント設定
        
        # 1着率のグラフ
        plt.figure(figsize=(10, 6))
        courses = list(stats['first_place']['percentage'].keys())
        percentages = list(stats['first_place']['percentage'].values())
        
        bars = plt.bar(courses, percentages, color=sns.color_palette("viridis", len(courses)))
        
        plt.title('ボートレース戸田 コース別1着率', fontsize=16)
        plt.xlabel('コース番号', fontsize=14)
        plt.ylabel('1着率 (%)', fontsize=14)
        plt.xticks(courses)
        plt.ylim(0, max(percentages) * 1.2)
        
        # 数値をバーの上に表示
        for bar, percentage in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{percentage}%', ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'course_first_place_rate.png'), dpi=300)
        plt.close()
        
        # 入着率のヒートマップ
        plt.figure(figsize=(12, 8))
        
        # データの準備
        heatmap_data = np.zeros((3, 6))
        for i, place in enumerate(['first_place', 'second_place', 'third_place']):
            for course in range(1, 7):
                if course in stats[place]['percentage']:
                    heatmap_data[i, course-1] = stats[place]['percentage'][course]
        
        # ヒートマップの作成
        ax = sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu',
                         xticklabels=[f'{i}コース' for i in range(1, 7)],
                         yticklabels=['1着率', '2着率', '3着率'])
        
        plt.title('ボートレース戸田 コース別入着率 (%)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'course_placement_heatmap.png'), dpi=300)
        plt.close()
        
        # 人気の3連単組み合わせ
        plt.figure(figsize=(12, 8))
        
        combinations = list(stats['top_combinations'].keys())
        counts = list(stats['top_combinations'].values())
        
        # 降順にソート
        sorted_indices = np.argsort(counts)[::-1]
        combinations = [combinations[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        bars = plt.barh(combinations, counts, color=sns.color_palette("viridis", len(combinations)))
        
        plt.title('ボートレース戸田 人気の3連単組み合わせ', fontsize=16)
        plt.xlabel('出現回数', fontsize=14)
        plt.ylabel('3連単組み合わせ', fontsize=14)
        
        # 数値をバーの右に表示
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{count}回', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'popular_trifecta_combinations.png'), dpi=300)
        plt.close()
        
        print(f"Graphs saved to {self.graphs_dir}")
    
    def analyze_seasonal_trends(self, df):
        """
        季節ごとのトレンドを分析する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            レース結果のデータフレーム
            
        Returns:
        --------
        dict
            季節別統計情報
        """
        if df is None or df.empty:
            return None
        
        # 日付列をdatetime型に変換
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # 月情報を追加
        df['month'] = df['date'].dt.month
        
        # 季節の定義
        seasons = {
            'spring': [3, 4, 5],    # 春: 3-5月
            'summer': [6, 7, 8],    # 夏: 6-8月
            'autumn': [9, 10, 11],  # 秋: 9-11月
            'winter': [12, 1, 2]    # 冬: 12-2月
        }
        
        # 季節情報を追加
        def get_season(month):
            for season, months in seasons.items():
                if month in months:
                    return season
            return None
        
        df['season'] = df['month'].apply(get_season)
        
        # 季節ごとのコース別1着率を計算
        seasonal_stats = {}
        
        for season in seasons.keys():
            season_df = df[df['season'] == season]
            
            if len(season_df) > 0:
                # 1着のコース別集計
                first_place_counts = season_df['first'].value_counts().sort_index()
                first_place_percentage = (first_place_counts / len(season_df) * 100).round(1)
                
                seasonal_stats[season] = {
                    'counts': first_place_counts.to_dict(),
                    'percentage': first_place_percentage.to_dict(),
                    'total_races': len(season_df)
                }
        
        # 結果をJSONファイルに保存
        output_file = os.path.join(self.results_dir, "seasonal_statistics.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(seasonal_stats, f, ensure_ascii=False, indent=4)
        print(f"Seasonal statistics saved to {output_file}")
        
        return seasonal_stats
    
    def plot_seasonal_trends(self, seasonal_stats):
        """
        季節ごとのトレンドをグラフ化する
        
        Parameters:
        -----------
        seasonal_stats : dict
            季節別統計情報
            
        Returns:
        --------
        None
        """
        if seasonal_stats is None:
            return
        
        # プロットのスタイル設定
        plt.style.use('ggplot')
        sns.set(font='IPAexGothic')  # 日本語フォント設定
        
        # 季節ごとのコース別1着率の比較グラフ
        plt.figure(figsize=(14, 8))
        
        # 季節の日本語名
        season_names = {
            'spring': '春 (3-5月)',
            'summer': '夏 (6-8月)',
            'autumn': '秋 (9-11月)',
            'winter': '冬 (12-2月)'
        }
        
        # バーの位置を調整するためのオフセット
        bar_width = 0.15
        offsets = np.arange(len(seasonal_stats)) - (len(seasonal_stats) - 1) / 2 * bar_width
        
        # コース番号
        courses = range(1, 7)
        
        # 季節ごとにバーをプロット
        for i, (season, stats) in enumerate(seasonal_stats.items()):
            percentages = [stats['percentage'].get(course, 0) for course in courses]
            plt.bar([x + offsets[i] * bar_width for x in courses], percentages, 
                   width=bar_width, label=season_names.get(season, season))
        
        plt.title('ボートレース戸田 季節別コース1着率', fontsize=16)
        plt.xlabel('コース番号', fontsize=14)
        plt.ylabel('1着率 (%)', fontsize=14)
        plt.xticks(courses)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'seasonal_first_place_rate.png'), dpi=300)
        plt.close()
        
        # 季節ごとのヒートマップ
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (season, stats) in enumerate(seasonal_stats.items()):
            # データの準備
            heatmap_data = np.zeros(6)
            for course in range(1, 7):
                if course in stats['percentage']:
                    heatmap_data[course-1] = stats['percentage'][course]
            
            # ヒートマップの作成
            sns.heatmap(heatmap_data.reshape(1, -1), annot=True, fmt='.1f', cmap='YlGnBu',
                       xticklabels=[f'{i}コース' for i in range(1, 7)],
                       yticklabels=['1着率 (%)'], ax=axes[i])
            
            axes[i].set_title(f'{season_names.get(season, season)} (n={stats["total_races"]})', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'seasonal_heatmaps.png'), dpi=300)
        plt.close()
        
        print(f"Seasonal trend graphs saved to {self.graphs_dir}")

# 使用例
if __name__ == "__main__":
    analyzer = BoatraceDataAnalyzer()
    
    # データの読み込み
    df = analyzer.load_data("data/toda_race_results.csv")
    
    if df is not None:
        # コース別統計情報の分析
        stats = analyzer.analyze_course_statistics(df)
        analyzer.plot_course_statistics(stats)
        
        # 季節別トレンドの分析
        seasonal_stats = analyzer.analyze_seasonal_trends(df)
        analyzer.plot_seasonal_trends(seasonal_stats)
