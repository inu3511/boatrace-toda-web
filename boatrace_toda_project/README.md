# ボートレース戸田予想ツール

ボートレース戸田の特性を考慮した予想ツールです。コース特性、選手成績、モーター性能、気象条件など様々な要素を分析し、機械学習を活用してレース結果を予測します。

## 主な機能

- 本日のレース予想
- 指定日・指定レースの予想
- 過去データの収集と分析
- 予測モデルの訓練と評価
- 予想結果の詳細表示

## インストール方法

### 必要環境

- Python 3.8以上
- インターネット接続環境

### クイックスタート

1. インストールスクリプトを実行します：

```bash
bash install.sh
```

2. ツールを起動します：

```bash
python main.py
```

詳細なインストール方法と使い方については、[ユーザーマニュアル](docs/user_manual.md)を参照してください。

## ディレクトリ構造

```
boatrace_toda_project/
├── data/                  # データ保存ディレクトリ
│   ├── raw/               # 生データ
│   └── processed/         # 処理済みデータ
├── docs/                  # ドキュメント
│   └── user_manual.md     # ユーザーマニュアル
├── logs/                  # ログファイル
├── models/                # 訓練済みモデル
├── results/               # 予測結果
│   └── graphs/            # 評価グラフ
├── src/                   # ソースコード
│   ├── data_collection/   # データ収集モジュール
│   ├── data_processing/   # データ処理モジュール
│   ├── prediction/        # 予測モジュール
│   └── ui/                # ユーザーインターフェース
├── tests/                 # テストコード
├── install.sh             # インストールスクリプト
├── main.py                # メインスクリプト
└── README.md              # 説明ファイル
```

## 使用方法

ツールを起動すると、以下のようなメインメニューが表示されます：

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ メインメニュー             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1: 本日のレース予想        │
│ 2: 指定日のレース予想      │
│ 3: データ収集と処理        │
│ 4: モデル訓練              │
│ 5: 全パイプライン実行      │
│ 0: 終了                    │
└────────────────────────────┘
```

初回使用時は、まず「3: データ収集と処理」を選択してデータを収集し、次に「4: モデル訓練」を選択してモデルを訓練してください。その後、「1: 本日のレース予想」または「2: 指定日のレース予想」を選択して予想結果を確認できます。

## 注意事項

- このツールはボートレース戸田の特性に特化して設計されています。
- 予想結果はあくまで参考情報であり、的中を保証するものではありません。
- 公式サイトのレイアウト変更により、データ収集が正常に機能しなくなる可能性があります。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

© 2025 ボートレース戸田予想ツール
