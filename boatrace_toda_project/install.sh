#!/bin/bash

# ボートレース戸田予想ツール インストールスクリプト

echo "ボートレース戸田予想ツール インストールを開始します..."

# 必要なライブラリをインストール
echo "必要なライブラリをインストールしています..."
pip install pandas numpy scikit-learn matplotlib seaborn joblib beautifulsoup4 requests rich

# ディレクトリ構造を確認
echo "ディレクトリ構造を確認しています..."
mkdir -p data/raw data/processed logs models results/graphs

# 権限を設定
echo "実行権限を設定しています..."
chmod +x main.py

echo "インストールが完了しました！"
echo "以下のコマンドでツールを起動できます："
echo "python main.py"
