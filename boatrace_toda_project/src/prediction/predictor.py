"""
ボートレース戸田の予測実行モジュール
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import argparse

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.collector import BoatraceDataCollector
from data_processing.preprocessor import BoatraceDataPreprocessor
from prediction.model import BoatracePredictionModel

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'predictor.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

class BoatracePredictor:
    """
    ボートレース戸田の予測を実行するクラス
    """
    
    def __init__(self, root_dir=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        root_dir : str, optional
            プロジェクトのルートディレクトリ
        """
        if root_dir is None:
            self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        else:
            self.root_dir = root_dir
        
        # 各ディレクトリの設定
        self.data_dir = os.path.join(self.root_dir, "data")
        self.models_dir = os.path.join(self.root_dir, "models")
        self.results_dir = os.path.join(self.root_dir, "results")
        self.logs_dir = os.path.join(self.root_dir, "logs")
        
        # 必要なディレクトリを作成
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 各モジュールのインスタンス化
        self.collector = BoatraceDataCollector(data_dir=self.data_dir)
        self.preprocessor = BoatraceDataPreprocessor(data_dir=self.data_dir)
        self.model = BoatracePredictionModel(
            data_dir=self.data_dir,
            models_dir=self.models_dir,
            results_dir=self.results_dir
        )
    
    def collect_and_process_data(self, start_date, end_date):
        """
        データの収集と処理を実行する
        
        Parameters:
        -----------
        start_date : str
            開始日（YYYYMMDD形式）
        end_date : str
            終了日（YYYYMMDD形式）
            
        Returns:
        --------
        pandas.DataFrame
            処理済みのデータフレーム
        """
        logger.info(f"Starting data collection and processing for period: {start_date} to {end_date}")
        
        # データ収集
        collected_data = self.collector.collect_data_for_period(start_date, end_date)
        
        if collected_data is None:
            logger.error("Data collection failed")
            return None
        
        # 収集したデータのファイルパス
        results_file = os.path.join(
            self.collector.processed_data_dir,
            f"toda_race_results_{start_date}_to_{end_date}.csv"
        )
        details_file = os.path.join(
            self.collector.raw_data_dir,
            f"toda_race_details_{start_date}_to_{end_date}.json"
        )
        motor_boat_file = os.path.join(
            self.collector.raw_data_dir,
            f"toda_motor_boat_info_{start_date}_to_{end_date}.json"
        )
        
        # ファイルの存在確認
        if not os.path.exists(results_file) or not os.path.exists(details_file):
            logger.error(f"Required files not found: {results_file} or {details_file}")
            return None
        
        # データ処理
        motor_boat_path = motor_boat_file if os.path.exists(motor_boat_file) else None
        processed_df = self.preprocessor.process_data(results_file, details_file, motor_boat_path)
        
        if processed_df is None:
            logger.error("Data processing failed")
            return None
        
        logger.info(f"Data collection and processing completed: {len(processed_df)} records")
        return processed_df
    
    def train_model(self, data_file=None):
        """
        予測モデルを訓練する
        
        Parameters:
        -----------
        data_file : str, optional
            訓練データファイルのパス
            
        Returns:
        --------
        dict
            訓練結果
        """
        logger.info("Starting model training")
        
        if data_file is None:
            # 最新の処理済みデータファイルを探す
            processed_files = [
                f for f in os.listdir(self.preprocessor.processed_data_dir)
                if f.startswith("toda_processed_data_") and f.endswith(".csv")
            ]
            
            if not processed_files:
                logger.error("No processed data files found")
                return None
            
            # 最新のファイルを使用
            processed_files.sort(reverse=True)
            data_file = os.path.join(self.preprocessor.processed_data_dir, processed_files[0])
        
        # モデルの訓練と評価
        results = self.model.train_and_evaluate(data_file)
        
        if results is None:
            logger.error("Model training failed")
            return None
        
        logger.info("Model training completed")
        return results
    
    def predict_race(self, date_str, race_num, model_name='random_forest'):
        """
        指定したレースの予測を実行する
        
        Parameters:
        -----------
        date_str : str
            日付（YYYYMMDD形式）
        race_num : int
            レース番号
        model_name : str, optional
            使用するモデル名
            
        Returns:
        --------
        dict
            予測結果
        """
        logger.info(f"Starting prediction for race {race_num} on {date_str}")
        
        try:
            # レース情報の取得
            race_details = self.collector.get_race_details(date_str, race_num)
            
            if race_details is None:
                logger.error(f"Failed to get race details for race {race_num} on {date_str}")
                return None
            
            # モーター・ボート情報の取得
            motor_boat_info = self.collector.get_motor_boat_info(date_str)
            
            # レース情報をデータフレームに変換
            race_data = []
            
            for racer in race_details.get('racer_info', []):
                racer_dict = {
                    'date': date_str,
                    'race_number': race_num,
                    'waku': racer['waku'],
                    'racer_no': racer['racer_no'],
                    'racer_name': racer['racer_name'],
                    'course': racer['course'],
                    'st_time': racer['st_time'],
                    'race_time': racer['race_time']
                }
                
                # 天候情報を追加
                for key, value in race_details.get('weather_info', {}).items():
                    racer_dict[f'weather_{key}'] = value
                
                # モーター・ボート情報を追加（あれば）
                if motor_boat_info is not None:
                    for motor_boat in motor_boat_info.get('motor_boat_info', []):
                        if motor_boat['motor_no'] == racer_dict.get('motor_no'):
                            racer_dict['boat_no'] = motor_boat['boat_no']
                            racer_dict['motor_2rate'] = motor_boat['motor_2rate'].replace('%', '')
                            racer_dict['motor_3rate'] = motor_boat['motor_3rate'].replace('%', '')
                            racer_dict['boat_2rate'] = motor_boat['boat_2rate'].replace('%', '')
                            racer_dict['boat_3rate'] = motor_boat['boat_3rate'].replace('%', '')
                            break
                
                race_data.append(racer_dict)
            
            if not race_data:
                logger.error(f"No valid race data for race {race_num} on {date_str}")
                return None
            
            race_df = pd.DataFrame(race_data)
            
            # データの前処理
            preprocessed_df = self.preprocessor.preprocess_data(race_df)
            
            if preprocessed_df is None:
                logger.error("Failed to preprocess race data")
                return None
            
            # 特徴量の作成
            featured_df = self.preprocessor.create_features(preprocessed_df)
            
            if featured_df is None:
                logger.error("Failed to create features for race data")
                return None
            
            # 予測の実行
            prediction_result = self.model.predict_race(featured_df, model_name)
            
            if prediction_result is None:
                logger.error("Failed to make prediction for race")
                return None
            
            # 予測結果の保存
            output_file = os.path.join(
                self.results_dir,
                f"toda_race_{date_str}_{race_num}_prediction.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(prediction_result, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Prediction completed and saved to {output_file}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting race: {e}")
            return None
    
    def predict_today_races(self, model_name='random_forest'):
        """
        本日のレースの予測を実行する
        
        Parameters:
        -----------
        model_name : str, optional
            使用するモデル名
            
        Returns:
        --------
        dict
            予測結果
        """
        today = datetime.now().strftime('%Y%m%d')
        logger.info(f"Starting prediction for today's races: {today}")
        
        try:
            # 本日のレース結果を取得（レース番号の一覧を取得するため）
            results = self.collector.get_race_results(today)
            
            if results is None or results.empty:
                logger.error(f"No races found for today: {today}")
                return None
            
            # 各レースの予測を実行
            predictions = {}
            
            for race_num in sorted(results['race_number'].unique()):
                prediction = self.predict_race(today, race_num, model_name)
                
                if prediction is not None:
                    predictions[str(race_num)] = prediction
            
            # 全予測結果の保存
            output_file = os.path.join(
                self.results_dir,
                f"toda_races_{today}_predictions.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=4)
            
            logger.info(f"All predictions completed and saved to {output_file}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting today's races: {e}")
            return None
    
    def run_full_pipeline(self, days_back=30, model_name='random_forest'):
        """
        データ収集から予測までの全パイプラインを実行する
        
        Parameters:
        -----------
        days_back : int, optional
            過去何日分のデータを収集するか
        model_name : str, optional
            使用するモデル名
            
        Returns:
        --------
        dict
            実行結果
        """
        logger.info(f"Starting full pipeline: collecting {days_back} days of data and making predictions")
        
        try:
            # 日付範囲の設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            
            # データの収集と処理
            processed_df = self.collect_and_process_data(start_date_str, end_date_str)
            
            if processed_df is None:
                logger.error("Full pipeline failed: data collection and processing failed")
                return {
                    'status': 'error',
                    'message': 'Data collection and processing failed'
                }
            
            # モデルの訓練
            training_results = self.train_model()
            
            if training_results is None:
                logger.error("Full pipeline failed: model training failed")
                return {
                    'status': 'error',
                    'message': 'Model training failed'
                }
            
            # 本日のレースの予測
            predictions = self.predict_today_races(model_name)
            
            if predictions is None:
                logger.error("Full pipeline failed: race prediction failed")
                return {
                    'status': 'error',
                    'message': 'Race prediction failed'
                }
            
            logger.info("Full pipeline completed successfully")
            
            return {
                'status': 'success',
                'data_collection': {
                    'start_date': start_date_str,
                    'end_date': end_date_str,
                    'records': len(processed_df)
                },
                'model_training': {
                    'models': list(training_results.keys()),
                    'best_model': max(training_results, key=lambda k: training_results[k]['accuracy'])
                },
                'predictions': {
                    'date': end_date_str,
                    'races': len(predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error running full pipeline: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

def main():
    """
    コマンドラインからの実行用メイン関数
    """
    parser = argparse.ArgumentParser(description='ボートレース戸田の予測ツール')
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # collect コマンド
    collect_parser = subparsers.add_parser('collect', help='データ収集を実行')
    collect_parser.add_argument('--start-date', type=str, required=True, help='開始日（YYYYMMDD形式）')
    collect_parser.add_argument('--end-date', type=str, required=True, help='終了日（YYYYMMDD形式）')
    
    # train コマンド
    train_parser = subparsers.add_parser('train', help='モデル訓練を実行')
    train_parser.add_argument('--data-file', type=str, help='訓練データファイルのパス')
    
    # predict コマンド
    predict_parser = subparsers.add_parser('predict', help='予測を実行')
    predict_parser.add_argument('--date', type=str, help='日付（YYYYMMDD形式）')
    predict_parser.add_argument('--race', type=int, help='レース番号')
    predict_parser.add_argument('--model', type=str, default='random_forest', help='使用するモデル名')
    predict_parser.add_argument('--today', action='store_true', help='本日の全レースを予測')
    
    # pipeline コマンド
    pipeline_parser = subparsers.add_parser('pipeline', help='全パイプラインを実行')
    pipeline_parser.add_argument('--days', type=int, default=30, help='過去何日分のデータを収集するか')
    pipeline_parser.add_argument('--model', type=str, default='random_forest', help='使用するモデル名')
    
    args = parser.parse_args()
    
    # ロガーの設定
    os.makedirs('logs', exist_ok=True)
    
    # 予測器のインスタンス化
    predictor = BoatracePredictor()
    
    # コマンドに応じた処理
    if args.command == 'collect':
        predictor.collect_and_process_data(args.start_date, args.end_date)
    
    elif args.command == 'train':
        predictor.train_model(args.data_file)
    
    elif args.command == 'predict':
        if args.today:
            predictor.predict_today_races(args.model)
        elif args.date and args.race:
            predictor.predict_race(args.date, args.race, args.model)
        else:
            parser.error("predict コマンドには --today フラグか、--date と --race の両方が必要です")
    
    elif args.command == 'pipeline':
        predictor.run_full_pipeline(args.days, args.model)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
