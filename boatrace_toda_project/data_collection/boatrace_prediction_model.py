import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class BoatracePredictionModel:
    """
    ボートレース戸田の予想モデルを構築・評価するクラス
    """
    
    def __init__(self, data_dir=None, models_dir=None):
        if data_dir is None:
            self.data_dir = os.path.join(os.getcwd(), "data")
        else:
            self.data_dir = data_dir
        
        if models_dir is None:
            self.models_dir = os.path.join(os.getcwd(), "models")
        else:
            self.models_dir = models_dir
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 結果の保存先ディレクトリ
        self.results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # グラフの保存先ディレクトリ
        self.graphs_dir = os.path.join(self.results_dir, "model_evaluation")
        os.makedirs(self.graphs_dir, exist_ok=True)
    
    def prepare_features(self, df):
        """
        予測モデルの特徴量を準備する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            レース結果のデータフレーム
            
        Returns:
        --------
        pandas.DataFrame
            特徴量のデータフレーム
        """
        if df is None or df.empty:
            return None
        
        # 日付列をdatetime型に変換
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # 月と曜日の情報を追加
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # 季節の定義
        def get_season(month):
            if month in [3, 4, 5]:
                return 0  # 春
            elif month in [6, 7, 8]:
                return 1  # 夏
            elif month in [9, 10, 11]:
                return 2  # 秋
            else:
                return 3  # 冬
        
        df['season'] = df['month'].apply(get_season)
        
        # レース番号を数値型に変換
        if 'race_number' in df.columns and not pd.api.types.is_numeric_dtype(df['race_number']):
            df['race_number'] = pd.to_numeric(df['race_number'])
        
        # 特徴量として使用する列を選択
        feature_columns = [
            'race_number', 'month', 'day_of_week', 'season'
        ]
        
        # 気象情報があれば追加
        weather_columns = [col for col in df.columns if col.startswith('weather_')]
        feature_columns.extend(weather_columns)
        
        # 選手情報があれば追加
        racer_columns = [col for col in df.columns if col.startswith('racer_')]
        feature_columns.extend(racer_columns)
        
        # モーター情報があれば追加
        motor_columns = [col for col in df.columns if col.startswith('motor_')]
        feature_columns.extend(motor_columns)
        
        # 特徴量として使用可能な列のみを選択
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            print("Warning: No usable feature columns found in the data")
            return None
        
        # 特徴量のデータフレームを作成
        X = df[available_columns].copy()
        
        # カテゴリ変数をダミー変数に変換
        categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
        if categorical_columns:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        return X
    
    def train_model(self, df, target_column='first', test_size=0.2, random_state=42):
        """
        予測モデルを訓練する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            レース結果のデータフレーム
        target_column : str
            予測対象の列名（デフォルトは'first'）
        test_size : float
            テストデータの割合
        random_state : int
            乱数シード
            
        Returns:
        --------
        dict
            訓練済みモデルと評価結果
        """
        if df is None or df.empty:
            return None
        
        # 特徴量の準備
        X = self.prepare_features(df)
        if X is None:
            return None
        
        # 目的変数
        if target_column not in df.columns:
            print(f"Error: Target column '{target_column}' not found in the data")
            return None
        
        y = df[target_column]
        
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # モデルの定義
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        }
        
        results = {}
        
        # 各モデルを訓練・評価
        for name, model in models.items():
            print(f"Training {name} model...")
            model.fit(X_train, y_train)
            
            # テストデータでの予測
            y_pred = model.predict(X_test)
            
            # 評価指標の計算
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"{name} model evaluation:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # 特徴量の重要度
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # モデルの保存
            model_path = os.path.join(self.models_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
            
            # 結果の保存
            results[name] = {
                'model_path': model_path,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'feature_importance': feature_importance.to_dict(),
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
        with open(results_path, 'w', encoding='utf-8') as f:
            # feature_importanceはDataFrameなのでJSONに変換できないため除外
            json_results = {k: {kk: vv for kk, vv in v.items() if kk != 'feature_importance'} 
                           for k, v in results.items()}
            json.dump(json_results, f, ensure_ascii=False, indent=4)
        print(f"Evaluation results saved to {results_path}")
        
        return results
    
    def load_model(self, model_name='random_forest'):
        """
        訓練済みモデルを読み込む
        
        Parameters:
        -----------
        model_name : str
            モデル名（'random_forest'または'gradient_boosting'）
            
        Returns:
        --------
        object
            読み込んだモデル
        """
        model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found")
            return None
        
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, input_data, model_name='random_forest'):
        """
        新しいデータに対して予測を行う
        
        Parameters:
        -----------
        input_data : pandas.DataFrame
            予測対象のデータ
        model_name : str
            使用するモデル名
            
        Returns:
        --------
        numpy.ndarray
            予測結果
        """
        # モデルの読み込み
        model = self.load_model(model_name)
        if model is None:
            return None
        
        # 特徴量の準備
        X = self.prepare_features(input_data)
        if X is None:
            return None
        
        # 予測
        try:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def evaluate_model_performance(self, results):
        """
        モデルのパフォーマンスを評価し、グラフ化する
        
        Parameters:
        -----------
        results : dict
            train_model()の戻り値
            
        Returns:
        --------
        None
        """
        if results is None:
            return
        
        # モデル比較のバープロット
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
        
        print(f"Model performance comparison graph saved to {self.graphs_dir}")

# 使用例
if __name__ == "__main__":
    predictor = BoatracePredictionModel()
    
    # データの読み込み
    df = pd.read_csv("data/toda_race_results.csv")
    
    if df is not None:
        # モデルの訓練と評価
        results = predictor.train_model(df, target_column='first')
        
        if results is not None:
            # モデルのパフォーマンス評価
            predictor.evaluate_model_performance(results)
            
            # 新しいデータに対する予測
            # 例: 最新の10レースに対する予測
            new_data = df.tail(10)
            predictions = predictor.predict(new_data)
            
            if predictions is not None:
                print("\nPredictions for the latest 10 races:")
                for i, (pred, actual) in enumerate(zip(predictions['predictions'], new_data['first'])):
                    print(f"Race {i+1}: Predicted winner: {pred}, Actual winner: {actual}")
