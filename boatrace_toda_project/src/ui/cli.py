"""
ボートレース戸田予想ツールのコマンドラインインターフェース
"""

import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.predictor import BoatracePredictor

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'cli.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

# リッチコンソールの設定
console = Console()

class BoatraceCliApp:
    """
    ボートレース戸田予想ツールのCLIアプリケーション
    """
    
    def __init__(self):
        """
        初期化メソッド
        """
        # プロジェクトのルートディレクトリを取得
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # 各ディレクトリの設定
        self.data_dir = os.path.join(self.root_dir, "data")
        self.results_dir = os.path.join(self.root_dir, "results")
        self.logs_dir = os.path.join(self.root_dir, "logs")
        
        # 必要なディレクトリを作成
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 予測器のインスタンス化
        self.predictor = BoatracePredictor(root_dir=self.root_dir)
    
    def show_welcome(self):
        """
        ウェルカムメッセージを表示する
        """
        console.print(Panel.fit(
            "[bold blue]ボートレース戸田 予想ツール[/bold blue]\n"
            "公式サイトのデータを基に、AIによる予想を提供します。",
            title="ようこそ",
            border_style="blue"
        ))
    
    def show_menu(self):
        """
        メインメニューを表示する
        
        Returns:
        --------
        str
            選択されたメニュー項目
        """
        menu_items = {
            "1": "本日のレース予想",
            "2": "指定日のレース予想",
            "3": "データ収集と処理",
            "4": "モデル訓練",
            "5": "全パイプライン実行",
            "0": "終了"
        }
        
        console.print(Panel.fit(
            "\n".join([f"[bold]{k}[/bold]: {v}" for k, v in menu_items.items()]),
            title="メインメニュー",
            border_style="cyan"
        ))
        
        choice = Prompt.ask("選択してください", choices=list(menu_items.keys()))
        return choice
    
    def predict_today_races(self):
        """
        本日のレース予想を実行する
        """
        console.print("[bold cyan]本日のレース予想を実行します...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("予想中...", total=None)
            
            # 本日のレース予想を実行
            predictions = self.predictor.predict_today_races()
            
            progress.update(task, completed=True)
        
        if predictions is None:
            console.print("[bold red]予想に失敗しました。詳細はログを確認してください。[/bold red]")
            return
        
        # 予想結果の表示
        console.print(f"[bold green]本日のレース予想が完了しました。{len(predictions)}レースの予想結果:[/bold green]")
        
        for race_num, prediction in predictions.items():
            self._display_race_prediction(prediction)
    
    def predict_specific_race(self):
        """
        指定日・指定レースの予想を実行する
        """
        # 日付の入力
        date_str = Prompt.ask(
            "日付を入力してください（YYYYMMDD形式）",
            default=datetime.now().strftime('%Y%m%d')
        )
        
        # レース番号の入力
        race_num = Prompt.ask(
            "レース番号を入力してください（1-12）",
            choices=[str(i) for i in range(1, 13)]
        )
        
        console.print(f"[bold cyan]{date_str}の{race_num}Rの予想を実行します...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("予想中...", total=None)
            
            # 指定レースの予想を実行
            prediction = self.predictor.predict_race(date_str, int(race_num))
            
            progress.update(task, completed=True)
        
        if prediction is None:
            console.print("[bold red]予想に失敗しました。詳細はログを確認してください。[/bold red]")
            return
        
        # 予想結果の表示
        console.print(f"[bold green]{date_str}の{race_num}Rの予想が完了しました:[/bold green]")
        self._display_race_prediction(prediction)
    
    def _display_race_prediction(self, prediction):
        """
        レース予想結果を表示する
        
        Parameters:
        -----------
        prediction : dict
            予想結果
        """
        # レース情報の表示
        race_info = prediction.get('race_info', {})
        date = race_info.get('date', '')
        race_number = race_info.get('race_number', '')
        weather_info = race_info.get('weather_info', {})
        
        weather_str = ", ".join([f"{k}: {v}" for k, v in weather_info.items()])
        
        console.print(Panel.fit(
            f"日付: {date}\nレース番号: {race_number}\n気象条件: {weather_str}",
            title="レース情報",
            border_style="blue"
        ))
        
        # 各艇の予想結果を表示
        boats_prediction = prediction.get('boats_prediction', [])
        
        if boats_prediction:
            table = Table(title="艇別予想")
            
            table.add_column("枠番", style="cyan")
            table.add_column("選手名", style="green")
            table.add_column("選手登録番号", style="blue")
            table.add_column("予想着順", style="magenta")
            table.add_column("1着確率", style="red")
            
            for boat in boats_prediction:
                table.add_row(
                    str(boat.get('waku', '')),
                    boat.get('racer_name', ''),
                    boat.get('racer_no', ''),
                    str(boat.get('predicted_rank', '')),
                    f"{boat.get('win_probability', 0) * 100:.1f}%"
                )
            
            console.print(table)
        
        # 3連単予想の表示
        top_trifecta = prediction.get('top_trifecta', [])
        
        if top_trifecta:
            table = Table(title="3連単予想（上位5組）")
            
            table.add_column("組み合わせ", style="cyan")
            table.add_column("確率", style="red")
            
            for i, trifecta in enumerate(top_trifecta[:5]):
                table.add_row(
                    trifecta.get('combination', ''),
                    f"{trifecta.get('probability', 0) * 100:.2f}%"
                )
            
            console.print(table)
        
        console.print("")
    
    def collect_and_process_data(self):
        """
        データ収集と処理を実行する
        """
        # 日付範囲の入力
        days_back = Prompt.ask(
            "過去何日分のデータを収集しますか？",
            default="30"
        )
        
        try:
            days_back = int(days_back)
            if days_back <= 0:
                raise ValueError("正の整数を入力してください")
        except ValueError:
            console.print("[bold red]有効な数値を入力してください[/bold red]")
            return
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        console.print(f"[bold cyan]{start_date_str}から{end_date_str}までのデータを収集・処理します...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("データ収集・処理中...", total=None)
            
            # データ収集と処理を実行
            processed_df = self.predictor.collect_and_process_data(start_date_str, end_date_str)
            
            progress.update(task, completed=True)
        
        if processed_df is None:
            console.print("[bold red]データ収集・処理に失敗しました。詳細はログを確認してください。[/bold red]")
            return
        
        console.print(f"[bold green]データ収集・処理が完了しました。{len(processed_df)}件のデータを処理しました。[/bold green]")
    
    def train_model(self):
        """
        モデル訓練を実行する
        """
        # 最新の処理済みデータファイルを探す
        processed_files = [
            f for f in os.listdir(self.predictor.preprocessor.processed_data_dir)
            if f.startswith("toda_processed_data_") and f.endswith(".csv")
        ]
        
        if not processed_files:
            console.print("[bold red]処理済みデータファイルが見つかりません。先にデータ収集・処理を実行してください。[/bold red]")
            return
        
        # 最新のファイルを使用
        processed_files.sort(reverse=True)
        latest_file = processed_files[0]
        
        use_latest = Confirm.ask(
            f"最新の処理済みデータファイル（{latest_file}）を使用しますか？",
            default=True
        )
        
        if not use_latest:
            # ファイル一覧を表示
            console.print("利用可能なデータファイル:")
            for i, file in enumerate(processed_files):
                console.print(f"[cyan]{i+1}[/cyan]: {file}")
            
            file_index = Prompt.ask(
                "使用するファイルの番号を入力してください",
                choices=[str(i+1) for i in range(len(processed_files))],
                default="1"
            )
            
            selected_file = processed_files[int(file_index) - 1]
        else:
            selected_file = latest_file
        
        data_file = os.path.join(self.predictor.preprocessor.processed_data_dir, selected_file)
        
        console.print(f"[bold cyan]モデル訓練を実行します（データファイル: {selected_file}）...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("モデル訓練中...", total=None)
            
            # モデル訓練を実行
            results = self.predictor.train_model(data_file)
            
            progress.update(task, completed=True)
        
        if results is None:
            console.print("[bold red]モデル訓練に失敗しました。詳細はログを確認してください。[/bold red]")
            return
        
        # 訓練結果の表示
        console.print("[bold green]モデル訓練が完了しました。評価結果:[/bold green]")
        
        table = Table(title="モデル評価結果")
        
        table.add_column("モデル", style="cyan")
        table.add_column("正解率", style="green")
        table.add_column("適合率", style="blue")
        table.add_column("再現率", style="magenta")
        table.add_column("F1スコア", style="red")
        
        for name, result in results.items():
            table.add_row(
                name.replace('_', ' ').title(),
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1_score']:.4f}"
            )
        
        console.print(table)
        
        # 特徴量重要度の表示
        for name, result in results.items():
            feature_importance = result['feature_importance']
            
            table = Table(title=f"{name.replace('_', ' ').title()}モデルの特徴量重要度（上位10件）")
            
            table.add_column("特徴量", style="cyan")
            table.add_column("重要度", style="green")
            
            for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance'])):
                if i >= 10:
                    break
                
                table.add_row(
                    feature,
                    f"{importance:.4f}"
                )
            
            console.print(table)
    
    def run_full_pipeline(self):
        """
        全パイプラインを実行する
        """
        # 日数の入力
        days_back = Prompt.ask(
            "過去何日分のデータを収集しますか？",
            default="30"
        )
        
        try:
            days_back = int(days_back)
            if days_back <= 0:
                raise ValueError("正の整数を入力してください")
        except ValueError:
            console.print("[bold red]有効な数値を入力してください[/bold red]")
            return
        
        # モデルの選択
        model_name = Prompt.ask(
            "使用するモデルを選択してください",
            choices=["random_forest", "gradient_boosting"],
            default="random_forest"
        )
        
        console.print(f"[bold cyan]全パイプラインを実行します（過去{days_back}日分、モデル: {model_name}）...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("パイプライン実行中...", total=None)
            
            # 全パイプラインを実行
            result = self.predictor.run_full_pipeline(days_back, model_name)
            
            progress.update(task, completed=True)
        
        if result is None or result.get('status') != 'success':
            console.print("[bold red]パイプライン実行に失敗しました。詳細はログを確認してください。[/bold red]")
            if result and 'message' in result:
                console.print(f"エラーメッセージ: {result['message']}")
            return
        
        # 実行結果の表示
        console.print("[bold green]全パイプラインが正常に完了しました。実行結果:[/bold green]")
        
        data_collection = result.get('data_collection', {})
        model_training = result.get('model_training', {})
        predictions = result.get('predictions', {})
        
        console.print(Panel.fit(
            f"データ収集期間: {data_collection.get('start_date', '')} から {data_collection.get('end_date', '')}\n"
            f"処理レコード数: {data_collection.get('records', 0)}\n"
            f"訓練モデル: {', '.join(model_training.get('models', []))}\n"
            f"最良モデル: {model_training.get('best_model', '')}\n"
            f"予測日: {predictions.get('date', '')}\n"
            f"予測レース数: {predictions.get('races', 0)}",
            title="パイプライン実行結果",
            border_style="green"
        ))
    
    def run(self):
        """
        アプリケーションを実行する
        """
        self.show_welcome()
        
        while True:
            choice = self.show_menu()
            
            if choice == "0":
                console.print("[bold blue]アプリケーションを終了します。お疲れ様でした！[/bold blue]")
                break
            
            elif choice == "1":
                self.predict_today_races()
            
            elif choice == "2":
                self.predict_specific_race()
            
            elif choice == "3":
                self.collect_and_process_data()
            
            elif choice == "4":
                self.train_model()
            
            elif choice == "5":
                self.run_full_pipeline()

def main():
    """
    メイン関数
    """
    app = BoatraceCliApp()
    app.run()

if __name__ == "__main__":
    main()
