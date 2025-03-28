# ボートレース戸田予想ツール設計書

## 1. 基本構造設計

### 1.1 全体アーキテクチャ

ボートレース戸田予想ツールは、以下の主要コンポーネントで構成されます：

```
BoatraceTodaPredictor/
├── data_collection/         # データ収集モジュール
│   ├── collector.py         # レースデータ収集クラス
│   └── scraper.py           # Webスクレイピング機能
├── data_processing/         # データ処理モジュール
│   ├── preprocessor.py      # データ前処理クラス
│   └── feature_engineer.py  # 特徴量エンジニアリングクラス
├── prediction/              # 予測モジュール
│   ├── model.py             # 予測モデルクラス
│   ├── evaluator.py         # モデル評価クラス
│   └── predictor.py         # 予測実行クラス
├── visualization/           # 可視化モジュール
│   ├── plotter.py           # グラフ作成クラス
│   └── dashboard.py         # ダッシュボード表示クラス
├── ui/                      # ユーザーインターフェースモジュール
│   ├── web_ui.py            # Web UI実装
│   └── cli.py               # コマンドラインインターフェース実装
├── utils/                   # ユーティリティモジュール
│   ├── config.py            # 設定管理
│   ├── logger.py            # ログ管理
│   └── helpers.py           # ヘルパー関数
├── models/                  # 保存済みモデルディレクトリ
├── data/                    # データ保存ディレクトリ
│   ├── raw/                 # 生データ
│   └── processed/           # 処理済みデータ
├── results/                 # 結果保存ディレクトリ
├── config/                  # 設定ファイルディレクトリ
├── main.py                  # メインエントリーポイント
└── requirements.txt         # 依存ライブラリリスト
```

### 1.2 データフロー

データの流れは以下のようになります：

1. **データ収集**：公式サイトからレース情報、選手情報、気象情報などを収集
2. **データ処理**：収集したデータの前処理と特徴量エンジニアリング
3. **予測**：処理したデータを使用して予測モデルを実行
4. **結果表示**：予測結果の可視化と表示

### 1.3 モジュール間の連携

モジュール間の連携は以下のように設計します：

```
[データ収集モジュール] → [データ処理モジュール] → [予測モジュール] → [可視化モジュール] → [UI モジュール]
                                                                      ↑
                                                                      |
                                                        [ユーティリティモジュール]
```

## 2. 入力インターフェース設計

### 2.1 入力データ項目

ユーザーから入力を受け付ける項目は以下の通りです：

#### 必須入力項目
- **レース日付**：予想対象の日付（YYYY-MM-DD形式）
- **レース場**：「戸田」固定（将来的に他のレース場も対応可能な設計）
- **レース番号**：予想対象のレース番号（1R〜12R）

#### オプション入力項目
- **選手情報**：各枠の選手名/登録番号（自動取得も可能）
- **モーター情報**：各枠のモーター番号（自動取得も可能）
- **ボート情報**：各枠のボート番号（自動取得も可能）
- **気象条件**：風向き、風速、天候など（自動取得も可能）

### 2.2 入力フォーム設計

#### Web UI版
- レスポンシブデザインのフォーム
- 日付選択にはカレンダーピッカー
- レース番号はドロップダウンメニュー
- 選手情報、モーター情報などは自動取得ボタン付きのテキストフィールド
- 気象条件は自動取得ボタン付きのドロップダウンメニュー

#### CLI版
- コマンドライン引数による指定
- 設定ファイルによる一括指定
- インタラクティブモードでの対話的入力

### 2.3 データ検証方法

入力データの検証は以下のように行います：

- **形式検証**：日付形式、数値範囲などの基本的な検証
- **存在検証**：指定された日付・レース番号のレースが存在するか確認
- **整合性検証**：選手情報とモーター情報の整合性確認
- **欠損値処理**：欠損値がある場合の自動補完または警告表示

## 3. 出力形式設計

### 3.1 予想結果表示

予想結果は以下の形式で表示します：

#### 基本情報表示
- レース日付、レース場、レース番号
- 各枠の選手名、モーター番号、ボート番号

#### 予想結果表示
- **着順予想**：1着〜6着の予想順位
- **確率情報**：各艇の1着確率、3連単上位組み合わせの確率
- **信頼度**：予想の信頼度（高・中・低）

### 3.2 視覚化設計

予想結果の視覚化には以下の要素を含めます：

- **確率バーチャート**：各艇の1着確率をバーチャートで表示
- **コース図**：予想される展開をコース図で表示
- **ヒートマップ**：3連単組み合わせの確率をヒートマップで表示
- **過去成績グラフ**：各選手の過去成績をレーダーチャートで表示

### 3.3 詳細情報表示

詳細情報として以下を表示します：

- **選手詳細**：各選手の戸田での成績、得意コース、ST平均など
- **モーター詳細**：モーターの2連率、調整状況など
- **予想根拠**：予想の根拠となった主要因子の説明
- **注意点**：予想に影響する特殊要因（天候変化など）の警告

## 4. 必要なライブラリ選定

### 4.1 データ処理ライブラリ

- **pandas**: データフレーム操作、データ分析
- **numpy**: 数値計算、配列操作
- **scikit-learn**: 機械学習モデル構築、評価
- **requests**: HTTP通信
- **BeautifulSoup4**: HTMLパース、Webスクレイピング

### 4.2 UI構築ライブラリ

#### Web UI
- **Flask**: Webアプリケーションフレームワーク
- **Bootstrap**: レスポンシブUIコンポーネント
- **Chart.js**: インタラクティブなグラフ描画
- **jQuery**: DOM操作、イベント処理

#### CLI
- **argparse**: コマンドライン引数処理
- **rich**: リッチテキスト表示、プログレスバー
- **prompt_toolkit**: インタラクティブプロンプト

### 4.3 可視化ライブラリ

- **matplotlib**: 基本的なグラフ描画
- **seaborn**: 統計データの可視化
- **plotly**: インタラクティブな可視化

### 4.4 その他のライブラリ

- **joblib**: モデルの保存・読み込み
- **pytest**: テスト自動化
- **logging**: ログ管理
- **configparser**: 設定ファイル管理

## 5. 実装計画

### 5.1 開発優先順位

1. データ収集モジュール
2. データ処理モジュール
3. 予測モジュール
4. CLI版ユーザーインターフェース
5. 可視化モジュール
6. Web UI版ユーザーインターフェース

### 5.2 開発スケジュール

- **フェーズ1**: コアモジュール（データ収集、処理、予測）の実装
- **フェーズ2**: CLI版インターフェースの実装
- **フェーズ3**: 可視化機能の実装
- **フェーズ4**: Web UI版インターフェースの実装
- **フェーズ5**: テストと最適化

### 5.3 テスト計画

- **ユニットテスト**: 各モジュールの機能テスト
- **統合テスト**: モジュール間の連携テスト
- **予測精度検証**: 過去のレース結果との比較による精度検証
- **ユーザビリティテスト**: 実際のユーザーによる使用感テスト

## 6. 拡張性と保守性

### 6.1 拡張性の考慮

- **他のレース場対応**: レース場ごとの特性を設定ファイルで管理
- **新しい予測モデル追加**: モデルインターフェースを統一し、新モデルの追加を容易に
- **APIサービス化**: 将来的にAPIとして提供できるよう設計

### 6.2 保守性の考慮

- **モジュール分割**: 機能ごとに明確に分割し、依存関係を最小化
- **設定の外部化**: ハードコーディングを避け、設定を外部ファイルに
- **ドキュメント整備**: コード内ドキュメントとユーザーマニュアルの整備
- **ログ管理**: 詳細なログ記録による問題追跡の容易化

## 7. セキュリティと倫理的配慮

### 7.1 セキュリティ対策

- **データ保護**: 個人情報の適切な取り扱い
- **アクセス制限**: Webスクレイピング時のアクセス頻度制限
- **エラーハンドリング**: 適切なエラーハンドリングによる情報漏洩防止

### 7.2 倫理的配慮

- **免責事項**: 予想結果の利用に関する免責事項の明記
- **公式情報の尊重**: 公式サイトの利用規約遵守
- **ギャンブル依存症への配慮**: 適切な注意喚起の表示

## 8. 今後の展望

### 8.1 機能拡張案

- **リアルタイム予想**: レース直前情報を反映したリアルタイム予想
- **SNS連携**: 予想結果のSNS共有機能
- **履歴管理**: ユーザーごとの予想履歴管理
- **コミュニティ機能**: ユーザー間の予想共有・討論機能

### 8.2 ビジネスモデル案

- **フリーミアムモデル**: 基本機能は無料、高度な分析は有料
- **サブスクリプションモデル**: 月額制の予想サービス
- **アフィリエイトモデル**: 関連サービスへの誘導によるアフィリエイト収入
