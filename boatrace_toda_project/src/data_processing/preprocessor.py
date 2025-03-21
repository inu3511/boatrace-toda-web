"""
ボートレース戸田のデータ処理モジュール
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'preprocessor.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

class BoatraceDataPreprocessor:
    """
    ボートレース戸田のデータを前処理するクラス
    """
    
    def __init__(self, data_dir=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        data_dir : str, optional
            データディレクトリ
        """
        # プロジェクトのルートディレクトリを取得
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        if data_dir is None:
            self.data_dir = os.path.join(self.root_dir, "data")
        else:
            self.data_dir = data_dir
        
        # 生データと処理済みデータのディレクトリを設定
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # ログディレクトリを作成
        self.logs_dir = os.path.join(self.root_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def load_race_results(self, file_path):
        """
        レース結果データを読み込む
        
        Parameters:
        -----------
        file_path : str
            レース結果ファイルのパス
            
        Returns:
        --------
        pandas.DataFrame
            レース結果のデータフレーム
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} race results from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading race results from {file_path}: {e}")
            return None
    
    def load_race_details(self, file_path):
        """
        レース詳細データを読み込む
        
        Parameters:
        -----------
        file_path : str
            レース詳細ファイルのパス
            
        Returns:
        --------
        list
            レース詳細のリスト
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} race details from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading race details from {file_path}: {e}")
            return None
    
    def load_motor_boat_info(self, file_path):
        """
        モーター・ボート情報を読み込む
        
        Parameters:
        -----------
        file_path : str
            モーター・ボート情報ファイルのパス
            
        Returns:
        --------
        list
            モーター・ボート情報のリスト
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} motor/boat info from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading motor/boat info from {file_path}: {e}")
            return None
    
    def merge_race_data(self, results_df, details_list, motor_boat_list):
        """
        レース結果、詳細情報、モーター・ボート情報を結合する
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            レース結果のデータフレーム
        details_list : list
            レース詳細のリスト
        motor_boat_list : list
            モーター・ボート情報のリスト
            
        Returns:
        --------
        pandas.DataFrame
            結合したデータフレーム
        """
        if results_df is None or details_list is None:
            logger.error("Cannot merge data: results_df or details_list is None")
            return None
        
        try:
            # レース詳細情報をデータフレームに変換
            details_data = []
            
            for detail in details_list:
                date = detail['date']
                race_number = detail['race_number']
                
                # 天候情報を取得
                weather_info = detail.get('weather_info', {})
                weather_dict = {
                    'date': date,
                    'race_number': race_number
                }
                
                for key, value in weather_info.items():
                    weather_dict[f'weather_{key}'] = value
                
                # 選手情報を取得
                for racer in detail.get('racer_info', []):
                    racer_dict = weather_dict.copy()
                    racer_dict.update({
                        'waku': racer['waku'],
                        'racer_no': racer['racer_no'],
                        'racer_name': racer['racer_name'],
                        'course': racer['course'],
                        'st_time': racer['st_time'],
                        'race_time': racer['race_time'],
                        'rank': racer['rank']
                    })
                    details_data.append(racer_dict)
            
            if not details_data:
                logger.warning("No valid details data found")
                return results_df
            
            details_df = pd.DataFrame(details_data)
            
            # モーター・ボート情報をデータフレームに変換
            if motor_boat_list is not None:
                motor_boat_data = []
                
                for info in motor_boat_list:
                    date = info['date']
                    
                    for motor_boat in info.get('motor_boat_info', []):
                        motor_boat_dict = {
                            'date': date,
                            'motor_no': motor_boat['motor_no'],
                            'boat_no': motor_boat['boat_no'],
                            'motor_2rate': motor_boat['motor_2rate'],
                            'motor_3rate': motor_boat['motor_3rate'],
                            'boat_2rate': motor_boat['boat_2rate'],
                            'boat_3rate': motor_boat['boat_3rate']
                        }
                        motor_boat_data.append(motor_boat_dict)
                
                if motor_boat_data:
                    motor_boat_df = pd.DataFrame(motor_boat_data)
                    
                    # 数値型に変換
                    for col in ['motor_2rate', 'motor_3rate', 'boat_2rate', 'boat_3rate']:
                        motor_boat_df[col] = motor_boat_df[col].str.replace('%', '').astype(float)
                    
                    # details_dfとmotor_boat_dfを結合
                    details_df = pd.merge(
                        details_df,
                        motor_boat_df,
                        on=['date', 'motor_no'],
                        how='left'
                    )
            
            # レース結果とレース詳細を結合
            merged_df = pd.merge(
                results_df,
                details_df,
                on=['date', 'race_number'],
                how='left'
            )
            
            logger.info(f"Successfully merged data: {len(merged_df)} rows")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging race data: {e}")
            return results_df
    
    def preprocess_data(self, df):
        """
        データを前処理する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            前処理対象のデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            前処理済みのデータフレーム
        """
        if df is None or df.empty:
            logger.error("Cannot preprocess data: df is None or empty")
            return None
        
        try:
            # 日付列をdatetime型に変換
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            # 月と曜日の情報を追加
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # 季節の定義
            def get_season(month):
                if month in [3, 4, 5]:
                    return 'spring'  # 春
                elif month in [6, 7, 8]:
                    return 'summer'  # 夏
                elif month in [9, 10, 11]:
                    return 'autumn'  # 秋
                else:
                    return 'winter'  # 冬
            
            df['season'] = df['month'].apply(get_season)
            
            # STタイムを数値型に変換
            if 'st_time' in df.columns:
                df['st_time'] = pd.to_numeric(df['st_time'], errors='coerce')
            
            # レースタイムを数値型に変換（秒単位）
            if 'race_time' in df.columns:
                def convert_race_time(time_str):
                    try:
                        if pd.isna(time_str) or time_str == '':
                            return np.nan
                        
                        parts = time_str.split('.')
                        if len(parts) != 2:
                            return np.nan
                        
                        seconds = float(parts[0])
                        milliseconds = float(parts[1]) / 10
                        
                        return seconds + milliseconds
                    except:
                        return np.nan
                
                df['race_time_seconds'] = df['race_time'].apply(convert_race_time)
            
            # 風向きを数値に変換
            if 'weather_風向' in df.columns:
                wind_direction_map = {
                    '北': 0,
                    '北東': 45,
                    '東': 90,
                    '南東': 135,
                    '南': 180,
                    '南西': 225,
                    '西': 270,
                    '北西': 315
                }
                
                df['wind_direction_degree'] = df['weather_風向'].map(wind_direction_map)
            
            # 風速を数値に変換
            if 'weather_風速' in df.columns:
                df['wind_speed'] = df['weather_風速'].str.extract(r'(\d+\.?\d*)').astype(float)
            
            # 波高を数値に変換
            if 'weather_波高' in df.columns:
                df['wave_height'] = df['weather_波高'].str.extract(r'(\d+\.?\d*)').astype(float)
            
            # 欠損値の処理
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            logger.info(f"Successfully preprocessed data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return df
    
    def create_features(self, df):
        """
        特徴量を作成する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            特徴量作成対象のデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量を追加したデータフレーム
        """
        if df is None or df.empty:
            logger.error("Cannot create features: df is None or empty")
            return None
        
        try:
            # コピーを作成
            df_features = df.copy()
            
            # コース番号と着順の関係を特徴量化
            if 'course' in df_features.columns and 'rank' in df_features.columns:
                # 数値型に変換
                df_features['rank_num'] = pd.to_numeric(df_features['rank'], errors='coerce')
                
                # コース別の着順平均
                course_rank_mean = df_features.groupby('course')['rank_num'].mean().reset_index()
                course_rank_mean.columns = ['course', 'course_rank_mean']
                
                df_features = pd.merge(df_features, course_rank_mean, on='course', how='left')
                
                # コース別の1着率
                df_features['is_first'] = (df_features['rank_num'] == 1).astype(int)
                course_first_rate = df_features.groupby('course')['is_first'].mean().reset_index()
                course_first_rate.columns = ['course', 'course_first_rate']
                
                df_features = pd.merge(df_features, course_first_rate, on='course', how='left')
            
            # 選手の成績を特徴量化
            if 'racer_no' in df_features.columns and 'rank' in df_features.columns:
                # 選手別の着順平均
                racer_rank_mean = df_features.groupby('racer_no')['rank_num'].mean().reset_index()
                racer_rank_mean.columns = ['racer_no', 'racer_rank_mean']
                
                df_features = pd.merge(df_features, racer_rank_mean, on='racer_no', how='left')
                
                # 選手別の1着率
                racer_first_rate = df_features.groupby('racer_no')['is_first'].mean().reset_index()
                racer_first_rate.columns = ['racer_no', 'racer_first_rate']
                
                df_features = pd.merge(df_features, racer_first_rate, on='racer_no', how='left')
                
                # 選手別のSTタイム平均（あれば）
                if 'st_time' in df_features.columns:
                    racer_st_mean = df_features.groupby('racer_no')['st_time'].mean().reset_index()
                    racer_st_mean.columns = ['racer_no', 'racer_st_mean']
                    
                    df_features = pd.merge(df_features, racer_st_mean, on='racer_no', how='left')
            
            # モーター性能を特徴量化
            if 'motor_no' in df_features.columns and 'rank' in df_features.columns:
                # モーター別の着順平均
                motor_rank_mean = df_features.groupby('motor_no')['rank_num'].mean().reset_index()
                motor_rank_mean.columns = ['motor_no', 'motor_rank_mean']
                
                df_features = pd.merge(df_features, motor_rank_mean, on='motor_no', how='left')
                
                # モーター別の1着率
                motor_first_rate = df_features.groupby('motor_no')['is_first'].mean().reset_index()
                motor_first_rate.columns = ['motor_no', 'motor_first_rate']
                
                df_features = pd.merge(df_features, motor_first_rate, on='motor_no', how='left')
            
            # 気象条件と成績の関係を特徴量化
            if 'wind_speed' in df_features.columns and 'rank' in df_features.columns:
                # 風速を3段階にカテゴリ化
                df_features['wind_speed_cat'] = pd.cut(
                    df_features['wind_speed'],
                    bins=[0, 3, 7, float('inf')],
                    labels=['weak', 'medium', 'strong']
                )
                
                # 風速カテゴリ別のコース別1着率
                wind_course_first_rate = df_features.groupby(['wind_speed_cat', 'course'])['is_first'].mean().reset_index()
                wind_course_first_rate.columns = ['wind_speed_cat', 'course', 'wind_course_first_rate']
                
                df_features = pd.merge(
                    df_features,
                    wind_course_first_rate,
                    on=['wind_speed_cat', 'course'],
                    how='left'
                )
            
            # 季節とコースの関係を特徴量化
            if 'season' in df_features.columns and 'course' in df_features.columns and 'rank' in df_features.columns:
                # 季節別のコース別1着率
                season_course_first_rate = df_features.groupby(['season', 'course'])['is_first'].mean().reset_index()
                season_course_first_rate.columns = ['season', 'course', 'season_course_first_rate']
                
                df_features = pd.merge(
                    df_features,
                    season_course_first_rate,
                    on=['season', 'course'],
                    how='left'
                )
            
            logger.info(f"Successfully created features: {len(df_features)} rows")
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return df
    
    def process_data(self, results_file, details_file, motor_boat_file=None):
        """
        データ処理のメイン処理
        
        Parameters:
        -----------
        results_file : str
            レース結果ファイルのパス
        details_file : str
            レース詳細ファイルのパス
        motor_boat_file : str, optional
            モーター・ボート情報ファイルのパス
            
        Returns:
        --------
        pandas.DataFrame
            処理済みのデータフレーム
        """
        # データの読み込み
        results_df = self.load_race_results(results_file)
        details_list = self.load_race_details(details_file)
        motor_boat_list = self.load_motor_boat_info(motor_boat_file) if motor_boat_file else None
        
        if results_df is None or details_list is None:
            logger.error("Cannot process data: failed to load required data")
            return None
        
        # データの結合
        merged_df = self.merge_race_data(results_df, details_list, motor_boat_list)
        
        if merged_df is None:
            logger.error("Cannot process data: failed to merge data")
            return None
        
        # データの前処理
        preprocessed_df = self.preprocess_data(merged_df)
        
        if preprocessed_df is None:
            logger.error("Cannot process data: failed to preprocess data")
            return None
        
        # 特徴量の作成
        featured_df = self.create_features(preprocessed_df)
        
        if featured_df is None:
            logger.error("Cannot process data: failed to create features")
            return None
        
        # 処理済みデータの保存
        output_file = os.path.join(
            self.processed_data_dir,
            f"toda_processed_data_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        featured_df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        return featured_df

# 使用例
if __name__ == "__main__":
    # ログディレクトリを作成
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs'), exist_ok=True)
    
    preprocessor = BoatraceDataPreprocessor()
    
    # サンプルファイルパス
    results_file = os.path.join(preprocessor.processed_data_dir, "toda_race_results_20240301_to_20240315.csv")
    details_file = os.path.join(preprocessor.raw_data_dir, "toda_race_details_20240301_to_20240315.json")
    motor_boat_file = os.path.join(preprocessor.raw_data_dir, "toda_motor_boat_info_20240301_to_20240315.json")
    
    # データ処理の実行
    if os.path.exists(results_file) and os.path.exists(details_file):
        motor_boat_path = motor_boat_file if os.path.exists(motor_boat_file) else None
        preprocessor.process_data(results_file, details_file, motor_boat_path)
    else:
        logger.error(f"Required files not found: {results_file} or {details_file}")
