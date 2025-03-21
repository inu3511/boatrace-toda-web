"""
ボートレース戸田予想ツールのテストスクリプト
"""

import os
import sys
import unittest
import pandas as pd
import json
from datetime import datetime, timedelta

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_collection.collector import BoatraceDataCollector
from src.data_processing.preprocessor import BoatraceDataPreprocessor
from src.prediction.model import BoatracePredictionModel
from src.prediction.predictor import BoatracePredictor

class TestBoatraceDataCollector(unittest.TestCase):
    """
    BoatraceDataCollectorのテストクラス
    """
    
    def setUp(self):
        """
        テスト前の準備
        """
        self.collector = BoatraceDataCollector()
        
        # テスト用の日付（最近の日付を使用）
        self.test_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        self.test_race_num = 1
    
    def test_get_race_results(self):
        """
        get_race_resultsメソッドのテスト
        """
        results = self.collector.get_race_results(self.test_date)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(results, "レース結果の取得に失敗しました")
        
        if results is not None and not results.empty:
            # 必要な列が含まれていることを確認
            required_columns = ['date', 'race_number', 'first', 'second', 'third']
            for col in required_columns:
                self.assertIn(col, results.columns, f"結果に{col}列が含まれていません")
    
    def test_get_race_details(self):
        """
        get_race_detailsメソッドのテスト
        """
        details = self.collector.get_race_details(self.test_date, self.test_race_num)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(details, "レース詳細の取得に失敗しました")
        
        if details is not None:
            # 必要なキーが含まれていることを確認
            required_keys = ['date', 'race_number', 'weather_info', 'racer_info']
            for key in required_keys:
                self.assertIn(key, details, f"詳細に{key}キーが含まれていません")
            
            # 選手情報が含まれていることを確認
            self.assertTrue(len(details['racer_info']) > 0, "選手情報が含まれていません")
    
    def test_get_motor_boat_info(self):
        """
        get_motor_boat_infoメソッドのテスト
        """
        motor_boat_info = self.collector.get_motor_boat_info(self.test_date)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(motor_boat_info, "モーター・ボート情報の取得に失敗しました")
        
        if motor_boat_info is not None:
            # 必要なキーが含まれていることを確認
            required_keys = ['date', 'motor_boat_info']
            for key in required_keys:
                self.assertIn(key, motor_boat_info, f"情報に{key}キーが含まれていません")
            
            # モーター・ボート情報が含まれていることを確認
            self.assertTrue(len(motor_boat_info['motor_boat_info']) > 0, "モーター・ボート情報が含まれていません")

class TestBoatraceDataPreprocessor(unittest.TestCase):
    """
    BoatraceDataPreprocessorのテストクラス
    """
    
    def setUp(self):
        """
        テスト前の準備
        """
        self.preprocessor = BoatraceDataPreprocessor()
        
        # テスト用のデータフレーム
        self.test_df = pd.DataFrame({
            'date': ['20240315', '20240315', '20240315'],
            'race_number': [1, 1, 1],
            'waku': [1, 2, 3],
            'racer_no': ['1234', '5678', '9012'],
            'racer_name': ['選手A', '選手B', '選手C'],
            'course': [1, 2, 3],
            'st_time': ['0.12', '0.15', '0.18'],
            'race_time': ['1.45', '1.50', '1.55'],
            'rank': ['1', '2', '3'],
            'weather_風向': ['北', '北', '北'],
            'weather_風速': ['3m', '3m', '3m'],
            'weather_波高': ['5cm', '5cm', '5cm']
        })
    
    def test_preprocess_data(self):
        """
        preprocess_dataメソッドのテスト
        """
        preprocessed_df = self.preprocessor.preprocess_data(self.test_df)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(preprocessed_df, "データの前処理に失敗しました")
        
        if preprocessed_df is not None and not preprocessed_df.empty:
            # 必要な列が追加されていることを確認
            added_columns = ['month', 'day_of_week', 'season']
            for col in added_columns:
                self.assertIn(col, preprocessed_df.columns, f"前処理後のデータに{col}列が含まれていません")
            
            # 数値型に変換されていることを確認
            numeric_columns = ['st_time', 'race_time_seconds', 'wind_direction_degree', 'wind_speed', 'wave_height']
            for col in numeric_columns:
                if col in preprocessed_df.columns:
                    self.assertTrue(pd.api.types.is_numeric_dtype(preprocessed_df[col]), f"{col}列が数値型に変換されていません")
    
    def test_create_features(self):
        """
        create_featuresメソッドのテスト
        """
        # 前処理済みデータを使用
        preprocessed_df = self.preprocessor.preprocess_data(self.test_df)
        
        if preprocessed_df is not None and not preprocessed_df.empty:
            featured_df = self.preprocessor.create_features(preprocessed_df)
            
            # 結果がNoneでないことを確認
            self.assertIsNotNone(featured_df, "特徴量の作成に失敗しました")
            
            if featured_df is not None and not featured_df.empty:
                # 特徴量が追加されていることを確認
                feature_columns = ['rank_num', 'is_first']
                for col in feature_columns:
                    self.assertIn(col, featured_df.columns, f"特徴量作成後のデータに{col}列が含まれていません")

class TestBoatracePredictionModel(unittest.TestCase):
    """
    BoatracePredictionModelのテストクラス
    """
    
    def setUp(self):
        """
        テスト前の準備
        """
        self.model = BoatracePredictionModel()
        
        # テスト用のデータフレーム
        self.test_df = pd.DataFrame({
            'date': ['20240315', '20240315', '20240315', '20240315', '20240315', '20240315'],
            'race_number': [1, 1, 1, 1, 1, 1],
            'waku': [1, 2, 3, 4, 5, 6],
            'racer_no': ['1234', '5678', '9012', '3456', '7890', '1357'],
            'racer_name': ['選手A', '選手B', '選手C', '選手D', '選手E', '選手F'],
            'course': [1, 2, 3, 4, 5, 6],
            'st_time': [0.12, 0.15, 0.18, 0.20, 0.22, 0.25],
            'race_time_seconds': [1.45, 1.50, 1.55, 1.60, 1.65, 1.70],
            'rank_num': [1, 2, 3, 4, 5, 6],
            'is_first': [1, 0, 0, 0, 0, 0],
            'month': [3, 3, 3, 3, 3, 3],
            'day_of_week': [4, 4, 4, 4, 4, 4],
            'season': ['spring', 'spring', 'spring', 'spring', 'spring', 'spring'],
            'wind_direction_degree': [0, 0, 0, 0, 0, 0],
            'wind_speed': [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            'wave_height': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            'first': [1, 0, 0, 0, 0, 0]  # 目的変数
        })
    
    def test_prepare_features(self):
        """
        prepare_featuresメソッドのテスト
        """
        X, y = self.model.prepare_features(self.test_df, target_column='first')
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(X, "特徴量の準備に失敗しました")
        self.assertIsNotNone(y, "目的変数の準備に失敗しました")
        
        if X is not None and y is not None:
            # 特徴量の列数が正しいことを確認
            self.assertTrue(X.shape[1] > 0, "特徴量の列数が0です")
            
            # 目的変数の長さが正しいことを確認
            self.assertEqual(len(y), len(self.test_df), "目的変数の長さがデータフレームの行数と一致しません")
    
    def test_train_model(self):
        """
        train_modelメソッドのテスト
        """
        # 特徴量と目的変数の準備
        X, y = self.model.prepare_features(self.test_df, target_column='first')
        
        if X is not None and y is not None and len(X) >= 5:  # 訓練に十分なデータがあるか確認
            results = self.model.train_model(X, y, test_size=0.2)
            
            # 結果がNoneでないことを確認
            self.assertIsNotNone(results, "モデルの訓練に失敗しました")
            
            if results is not None:
                # 必要なモデルが含まれていることを確認
                required_models = ['random_forest', 'gradient_boosting']
                for model_name in required_models:
                    self.assertIn(model_name, results, f"訓練結果に{model_name}モデルが含まれていません")
                
                # 評価指標が含まれていることを確認
                for model_name, result in results.items():
                    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                    for metric in required_metrics:
                        self.assertIn(metric, result, f"{model_name}モデルの結果に{metric}指標が含まれていません")

class TestBoatracePredictor(unittest.TestCase):
    """
    BoatracePredictorのテストクラス
    """
    
    def setUp(self):
        """
        テスト前の準備
        """
        self.predictor = BoatracePredictor()
        
        # テスト用の日付（最近の日付を使用）
        self.test_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        self.test_race_num = 1
    
    def test_predict_race(self):
        """
        predict_raceメソッドのテスト
        """
        # モデルファイルが存在しない場合はスキップ
        model_path = os.path.join(self.predictor.models_dir, "random_forest_model.joblib")
        if not os.path.exists(model_path):
            self.skipTest("モデルファイルが存在しないためテストをスキップします")
        
        prediction = self.predictor.predict_race(self.test_date, self.test_race_num)
        
        # 結果がNoneでないことを確認
        self.assertIsNotNone(prediction, "レース予測に失敗しました")
        
        if prediction is not None:
            # 必要なキーが含まれていることを確認
            required_keys = ['race_info', 'boats_prediction', 'top_trifecta']
            for key in required_keys:
                self.assertIn(key, prediction, f"予測結果に{key}キーが含まれていません")
            
            # 艇別予測が含まれていることを確認
            self.assertTrue(len(prediction['boats_prediction']) > 0, "艇別予測が含まれていません")
            
            # 3連単予測が含まれていることを確認
            self.assertTrue(len(prediction['top_trifecta']) > 0, "3連単予測が含まれていません")

def run_tests():
    """
    全テストを実行する
    """
    # テストスイートの作成
    suite = unittest.TestSuite()
    
    # テストケースの追加
    suite.addTest(unittest.makeSuite(TestBoatraceDataCollector))
    suite.addTest(unittest.makeSuite(TestBoatraceDataPreprocessor))
    suite.addTest(unittest.makeSuite(TestBoatracePredictionModel))
    suite.addTest(unittest.makeSuite(TestBoatracePredictor))
    
    # テストの実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    run_tests()
