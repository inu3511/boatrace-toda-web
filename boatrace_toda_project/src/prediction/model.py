"""
ボートレース戸田の予測モデルモジュール
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'model.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

class BoatracePredictionModel:
    """
    ボートレース戸田の予測モデルを構築・評価するクラス
    """
    
    def __init__(self, data_dir=None, models_dir=None, results_dir=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        data_dir : str, optional
            データディレクトリ
        models_dir : str, optional
            モデル保存ディレクトリ
        results_dir : str, optional
            結果保存ディレクトリ
        """
        # プロジェクトのルートディレクトリを取得
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        if data_dir is None:
            self.data_dir = os.path.join(self.root_dir, "data")
        else:
            self.data_dir = data_dir
        
        if models_dir is None:
            self.models_dir = os.path.join(self.root_dir, "models")
        else:
            self.models_dir = models_dir
        
        if results_dir is None:
            self.results_dir = os.path.join(self.root_dir, "results")
        else:
            self.results_dir = results_dir
        
        # 処理済みデータのディレクトリを設定
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        
        # 必要なディレクトリを作成
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # グラフの保存先ディレクトリ
        self.graphs_dir = os.path.join(self.results_dir, "graphs")
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        # ログディレクトリを作成
        self.logs_dir = os.path.join(self.root_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # モデルの設定
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
    
    def load_data(self, file_path):
        """
        処理済みデータを読み込む
        
        Parameters:
        -----------
        file_path : str
            データファイルのパス
            
        Returns:
        --------
        pandas.DataFrame
            読み込んだデータフレーム
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None
    
    def prepare_features(self, df, target_column='first'):
        """
        特徴量と目的変数を準備する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            データフレーム
        target_column : str, optional
            目的変数の列名
            
        Returns:
        --------
        tuple
            (特徴量のデータフレーム, 目的変数のシリーズ)
        """
        if df is None or df.empty:
            logger.error("Cannot prepare features: df is None or empty")
            return None, None
        
        try:
            # 目的変数が存在するか確認
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found in the data")
                return None, None
            
            # 特徴量として使用する列を選択
            feature_columns = [
                'course', 'waku', 'month', 'day_of_week', 'season',
                'st_time', 'race_time_seconds',
                'wind_direction_degree', 'wind_speed', 'wave_height',
                'course_rank_mean', 'course_first_rate',
                'racer_rank_mean', 'racer_first_rate', 'racer_st_mean',
                'motor_rank_mean', 'motor_first_rate',
                'wind_course_first_rate', 'season_course_first_rate'
            ]
            
            # 存在する列のみを使用
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if not available_columns:
                logger.error("No usable feature columns found in the data")
                return None, None
            
            # 特徴量のデータフレームを作成
            X = df[available_columns].copy()
            
            # 目的変数
            y = df[target_column]
            
            # カテゴリ変数をダミー変数に変換
            categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
            if categorical_columns:
                X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            
            # 欠損値の処理
            X = X.fillna(X.median())
            
            logger.info(f"Prepared features: {X.shape[1]} features, {len(X)} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        モデルを訓練する
        
        Parameters:
        -----------
        X : pandas.DataFrame
            特徴量のデータフレーム
        y : pandas.Series
            目的変数のシリーズ
        test_size : float, optional
            テストデータの割合
        random_state : int, optional
            乱数シード
            
        Returns:
        --------
        dict
            訓練結果
        """
        if X is None or y is None:
            logger.error("Cannot train model: X or y is None")
            return None
        
        try:
            # 訓練データとテストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            results = {}
            
            # 各モデルを訓練・評価
            for name, model in self.models.items():
                logger.info(f"Training {name} model...")
                model.fit(X_train, y_train)
                
                # テストデータでの予測
                y_pred = model.predict(X_test)
                
                # 評価指標の計算
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                logger.info(f"{name} model evaluation:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1 Score: {f1:.4f}")
                
                # 特徴量の重要度
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # モデルの保存
                model_path = os.path.join(self.models_dir, f"{name}_model.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # 結果の保存
                results[name] = {
                    'model_path': model_path,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'feature_importance': feature_importance,
                    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 特徴量の重要度をグラフ化
                plt.figure(figsize=(10, 8))
                top_features = feature_importance.head(15)
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title(f'Top 15 Feature Importance - {name.replace("_", " ").title()}', fontsize=16)
                plt.xlabel('Importance', fontsize=14)
                plt.ylabel('Feature', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.graphs_dir, f"{name}_feature_importance.png"), dpi=300)
                plt.close()
            
            # 結果をJSONファイルに保存
            results_path = os.path.join(self.results_dir, "model_evaluation_results.json")
            
            # DataFrameはJSON変換できないため、辞書に変換
            json_results = {}
            for name, result in results.items():
                json_results[name] = {
                    'model_path': result['model_path'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'feature_importance': result['feature_importance'].to_dict(),
                    'training_date': result['training_date']
                }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Evaluation results saved to {results_path}")
            
            # モデル比較のバープロット
            self._plot_model_comparison(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def _plot_model_comparison(self, results):
        """
        モデル比較のグラフを作成する
        
        Parameters:
        -----------
        results : dict
            訓練結果
        """
        try:
            plt.figure(figsize=(12, 8))
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            model_names = list(results.keys())
            
            x = np.arange(len(metrics))
            width = 0.35
            
            for i, model_name in enumerate(model_names):
                values = [results[model_name][metric] for metric in metrics]
                plt.bar(x + i*width, values, width, label=model_name.replace('_', ' ').title())
            
            plt.xlabel('Metrics', fontsize=14)
            plt.ylabel('Score', fontsize=14)
            plt.title('Model Performance Comparison', fontsize=16)
            plt.xticks(x + width/2, [m.replace('_', ' ').title() for m in metrics])
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.graphs_dir, "model_performance_comparison.png"), dpi=300)
            plt.close()
            
            logger.info(f"Model comparison graph saved to {self.graphs_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting model comparison: {e}")
    
    def load_model(self, model_name='random_forest'):
        """
        保存済みモデルを読み込む
        
        Parameters:
        -----------
        model_name : str, optional
            モデル名
            
        Returns:
        --------
        object
            読み込んだモデル
        """
        model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict(self, X, model_name='random_forest'):
        """
        予測を実行する
        
        Parameters:
        -----------
        X : pandas.DataFrame
            特徴量のデータフレーム
        model_name : str, optional
            モデル名
            
        Returns:
        --------
        dict
            予測結果
        """
        if X is None or X.empty:
            logger.error("Cannot predict: X is None or empty")
            return None
        
        try:
            # モデルの読み込み
            model = self.load_model(model_name)
            
            if model is None:
                logger.error(f"Failed to load model: {model_name}")
                return None
            
            # 予測
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            logger.info(f"Predictions made for {len(X)} samples")
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'classes': model.classes_
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def predict_race(self, race_data, model_name='random_forest'):
        """
        レースの予測を実行する
        
        Parameters:
        -----------
        race_data : pandas.DataFrame
            レースデータ
        model_name : str, optional
            モデル名
            
        Returns:
        --------
        dict
            予測結果
        """
        if race_data is None or race_data.empty:
            logger.error("Cannot predict race: race_data is None or empty")
            return None
        
        try:
            # 特徴量の準備
            X, _ = self.prepare_features(race_data)
            
            if X is None:
                logger.error("Failed to prepare features for race prediction")
                return None
            
            # 予測
            prediction_result = self.predict(X, model_name)
            
            if prediction_result is None:
                logger.error("Failed to make predictions for race")
                return None
            
            # 予測結果を整形
            predictions = prediction_result['predictions']
            probabilities = prediction_result['probabilities']
            classes = prediction_result['classes']
            
            # 各艇の予測結果
            boats_prediction = []
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                boat_info = race_data.iloc[i].to_dict()
                
                # 確率情報を追加
                prob_dict = {str(cls): float(p) for cls, p in zip(classes, prob)}
                
                boats_prediction.append({
                    'waku': boat_info.get('waku', i+1),
                    'racer_no': boat_info.get('racer_no', ''),
                    'racer_name': boat_info.get('racer_name', ''),
                    'predicted_rank': int(pred),
                    'win_probability': prob_dict.get('1', 0.0),
                    'probabilities': prob_dict
                })
            
            # 勝率順にソート
            boats_prediction.sort(key=lambda x: x['win_probability'], reverse=True)
            
            # 3連単の組み合わせと確率を計算
            trifecta_combinations = []
            
            for i in range(len(boats_prediction)):
                for j in range(len(boats_prediction)):
                    if i == j:
                        continue
                    
                    for k in range(len(boats_prediction)):
                        if i == k or j == k:
                            continue
                        
                        first_boat = boats_prediction[i]
                        second_boat = boats_prediction[j]
                        third_boat = boats_prediction[k]
                        
                        # 3連単の確率（簡易計算）
                        probability = (
                            first_boat['win_probability'] *
                            second_boat['probabilities'].get('2', 0.1) *
                            third_boat['probabilities'].get('3', 0.1)
                        )
                        
                        trifecta_combinations.append({
                            'combination': f"{first_boat['waku']}-{second_boat['waku']}-{third_boat['waku']}",
                            'probability': probability
                        })
            
            # 確率順にソート
            trifecta_combinations.sort(key=lambda x: x['probability'], reverse=True)
            
            # 上位10件のみを返す
            top_trifecta = trifecta_combinations[:10]
            
            race_info = {
                'date': race_data['date'].iloc[0] if 'date' in race_data.columns else '',
                'race_number': race_data['race_number'].iloc[0] if 'race_number' in race_data.columns else '',
                'weather_info': {
                    col.replace('weather_', ''): race_data[col].iloc[0]
                    for col in race_data.columns if col.startswith('weather_')
                }
            }
            
            result = {
                'race_info': race_info,
                'boats_prediction': boats_prediction,
                'top_trifecta': top_trifecta
            }
            
            logger.info(f"Race prediction completed for race {race_info['race_number']} on {race_info['date']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting race: {e}")
            return None
    
    def train_and_evaluate(self, data_file, target_column='first'):
        """
        モデルの訓練と評価を実行する
        
        Parameters:
        -----------
        data_file : str
            データファイルのパス
        target_column : str, optional
            目的変数の列名
            
        Returns:
        --------
        dict
            訓練・評価結果
        """
        # データの読み込み
        df = self.load_data(data_file)
        
        if df is None:
            logger.error(f"Failed to load data from {data_file}")
            return None
        
        # 特徴量と目的変数の準備
        X, y = self.prepare_features(df, target_column)
        
        if X is None or y is None:
            logger.error("Failed to prepare features and target")
            return None
        
        # モデルの訓練と評価
        results = self.train_model(X, y)
        
        if results is None:
            logger.error("Failed to train and evaluate models")
            return None
        
        return results

# 使用例
if __name__ == "__main__":
    # ログディレクトリを作成
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs'), exist_ok=True)
    
    model = BoatracePredictionModel()
    
    # サンプルデータファイル
    data_file = os.path.join(model.processed_data_dir, "toda_processed_data_20240320.csv")
    
    # モデルの訓練と評価
    if os.path.exists(data_file):
        model.train_and_evaluate(data_file)
    else:
        logger.error(f"Data file not found: {data_file}")
