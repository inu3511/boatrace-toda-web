"""
ボートレース戸田予想ツールの統合テストスクリプト
"""

import os
import sys
import unittest
import pandas as pd
import json
from datetime import datetime, timedelta
import logging

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction.predictor import BoatracePredictor

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'integration_test.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

class TestBoatracePredictorIntegration(unittest.TestCase):
    """
    BoatracePredictorの統合テストクラス
    """
    
    def setUp(self):
        """
        テスト前の準備
        """
        self.predictor = BoatracePredictor()
        
        # テスト用の日付（最近の日付を使用）
        self.test_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        self.test_race_num = 1
        
        # テスト用の期間（短い期間を使用）
        self.test_days = 3
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.test_days)
        self.start_date_str = self.start_date.strftime('%Y%m%d')
        self.end_date_str = self.end_date.strftime('%Y%m%d')
    
    def test_collect_and_process_data(self):
        """
        collect_and_process_dataメソッドの統合テスト
        """
        logger.info(f"データ収集・処理の統合テストを開始します（期間: {self.start_date_str}～{self.end_date_str}）")
        
        processed_df = self.predictor.collect_and_process_data(self.start_date_str, self.end_date_str)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(processed_df, "データ収集・処理に失敗しました")
        
        if processed_df is not None and not processed_df.empty:
            logger.info(f"データ収集・処理が成功しました（{len(processed_df)}件のデータ）")
            
            # 必要な列が含まれていることを確認
            required_columns = ['date', 'race_number', 'waku', 'racer_no', 'course']
            for col in required_columns:
                self.assertIn(col, processed_df.columns, f"処理済みデータに{col}列が含まれていません")
            
            # 特徴量が含まれていることを確認
            feature_columns = ['month', 'day_of_week', 'season', 'is_first']
            for col in feature_columns:
                self.assertIn(col, processed_df.columns, f"処理済みデータに特徴量{col}が含まれていません")
    
    def test_train_model(self):
        """
        train_modelメソッドの統合テスト
        """
        logger.info("モデル訓練の統合テストを開始します")
        
        # 処理済みデータファイルを探す
        processed_files = [
            f for f in os.listdir(self.predictor.preprocessor.processed_data_dir)
            if f.startswith("toda_processed_data_") and f.endswith(".csv")
        ]
        
        if not processed_files:
            self.skipTest("処理済みデータファイルが見つからないためテストをスキップします")
        
        # 最新のファイルを使用
        processed_files.sort(reverse=True)
        data_file = os.path.join(self.predictor.preprocessor.processed_data_dir, processed_files[0])
        
        logger.info(f"モデル訓練に使用するデータファイル: {data_file}")
        
        results = self.predictor.train_model(data_file)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(results, "モデル訓練に失敗しました")
        
        if results is not None:
            logger.info("モデル訓練が成功しました")
            
            # 必要なモデルが含まれていることを確認
            required_models = ['random_forest', 'gradient_boosting']
            for model_name in required_models:
                self.assertIn(model_name, results, f"訓練結果に{model_name}モデルが含まれていません")
            
            # モデルファイルが作成されていることを確認
            for model_name in required_models:
                model_path = os.path.join(self.predictor.models_dir, f"{model_name}_model.joblib")
                self.assertTrue(os.path.exists(model_path), f"モデルファイル{model_path}が作成されていません")
    
    def test_predict_race(self):
        """
        predict_raceメソッドの統合テスト
        """
        logger.info(f"レース予測の統合テストを開始します（日付: {self.test_date}, レース: {self.test_race_num}）")
        
        # モデルファイルが存在しない場合はスキップ
        model_path = os.path.join(self.predictor.models_dir, "random_forest_model.joblib")
        if not os.path.exists(model_path):
            self.skipTest("モデルファイルが存在しないためテストをスキップします")
        
        prediction = self.predictor.predict_race(self.test_date, self.test_race_num)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(prediction, "レース予測に失敗しました")
        
        if prediction is not None:
            logger.info("レース予測が成功しました")
            
            # 必要なキーが含まれていることを確認
            required_keys = ['race_info', 'boats_prediction', 'top_trifecta']
            for key in required_keys:
                self.assertIn(key, prediction, f"予測結果に{key}キーが含まれていません")
            
            # 艇別予測が含まれていることを確認
            self.assertTrue(len(prediction['boats_prediction']) > 0, "艇別予測が含まれていません")
            
            # 3連単予測が含まれていることを確認
            self.assertTrue(len(prediction['top_trifecta']) > 0, "3連単予測が含まれていません")
            
            # 予測結果ファイルが作成されていることを確認
            result_file = os.path.join(
                self.predictor.results_dir,
                f"toda_race_{self.test_date}_{self.test_race_num}_prediction.json"
            )
            self.assertTrue(os.path.exists(result_file), f"予測結果ファイル{result_file}が作成されていません")
    
    def test_run_full_pipeline(self):
        """
        run_full_pipelineメソッドの統合テスト
        """
        logger.info(f"全パイプラインの統合テストを開始します（過去{self.test_days}日分）")
        
        result = self.predictor.run_full_pipeline(self.test_days)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(result, "全パイプライン実行に失敗しました")
        
        if result is not None:
            # ステータスが成功であることを確認
            self.assertEqual(result.get('status'), 'success', f"パイプライン実行のステータスが'success'ではありません: {result.get('status')}")
            
            if result.get('status') == 'success':
                logger.info("全パイプラインが正常に完了しました")
                
                # 必要なキーが含まれていることを確認
                required_keys = ['data_collection', 'model_training', 'predictions']
                for key in required_keys:
                    self.assertIn(key, result, f"実行結果に{key}キーが含まれていません")

def run_integration_tests():
    """
    統合テストを実行する
    """
    # テストスイートの作成
    suite = unittest.TestSuite()
    
    # テストケースの追加
    suite.addTest(unittest.makeSuite(TestBoatracePredictorIntegration))
    
    # テストの実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    run_integration_tests()
