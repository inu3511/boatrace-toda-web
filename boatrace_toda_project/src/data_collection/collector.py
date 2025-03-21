"""
ボートレース戸田のデータ収集モジュール
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
import json

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'collector.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

class BoatraceDataCollector:
    """
    ボートレース戸田のデータを収集するクラス
    """
    
    def __init__(self, data_dir=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        data_dir : str, optional
            データ保存ディレクトリ
        """
        # プロジェクトのルートディレクトリを取得
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        if data_dir is None:
            self.data_dir = os.path.join(self.root_dir, "data")
        else:
            self.data_dir = data_dir
        
        # 生データと処理済みデータのディレクトリを作成
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # ログディレクトリを作成
        self.logs_dir = os.path.join(self.root_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 設定
        self.base_url = "https://www.boatrace.jp/owpc/pc/race/"
        self.toda_jcd = "02"  # 戸田競艇場のコード
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_race_results(self, date_str, race_num=None):
        """
        指定した日付のレース結果を取得する
        
        Parameters:
        -----------
        date_str : str
            日付（YYYYMMDD形式）
        race_num : int, optional
            レース番号（指定しない場合は全レース）
            
        Returns:
        --------
        pandas.DataFrame
            レース結果のデータフレーム
        """
        url = f"{self.base_url}resultlist?jcd={self.toda_jcd}&hd={date_str}"
        logger.info(f"Fetching race results for date: {date_str}, URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # レース結果テーブルを取得
            result_tables = soup.select('table.is-w495')
            
            if not result_tables:
                logger.warning(f"No race results found for date: {date_str}")
                return None
            
            all_results = []
            
            for table in result_tables:
                # レース番号を取得
                race_number = int(table.find_previous('h3').text.strip().replace('R', ''))
                
                if race_num is not None and race_number != race_num:
                    continue
                
                # 3連単の結果を取得
                trifecta_row = table.select('tr')[1]
                trifecta_cells = trifecta_row.select('td')
                
                if len(trifecta_cells) >= 3:
                    # 組み合わせを取得
                    combination = trifecta_cells[1].text.strip()
                    first, second, third = map(int, combination.split('-'))
                    
                    # 払戻金を取得
                    payout = trifecta_cells[2].text.strip().replace('¥', '').replace(',', '')
                    payout = int(payout) if payout.isdigit() else 0
                    
                    result = {
                        'date': date_str,
                        'race_number': race_number,
                        'first': first,
                        'second': second,
                        'third': third,
                        'trifecta_payout': payout
                    }
                    
                    all_results.append(result)
            
            if all_results:
                df = pd.DataFrame(all_results)
                logger.info(f"Successfully fetched {len(df)} race results for date: {date_str}")
                return df
            else:
                logger.warning(f"No valid race results found for date: {date_str}")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching race results for date {date_str}: {e}")
            return None
    
    def get_race_details(self, date_str, race_num):
        """
        指定した日付・レース番号の詳細情報を取得する
        
        Parameters:
        -----------
        date_str : str
            日付（YYYYMMDD形式）
        race_num : int
            レース番号
            
        Returns:
        --------
        dict
            レース詳細情報
        """
        url = f"{self.base_url}raceresult?rno={race_num}&jcd={self.toda_jcd}&hd={date_str}"
        logger.info(f"Fetching race details for date: {date_str}, race: {race_num}, URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 天候・風向・風速・波高を取得
            weather_info = {}
            weather_table = soup.select_one('div.weather1')
            
            if weather_table:
                weather_items = weather_table.select('div.weather1_bodyUnitLabel')
                weather_values = weather_table.select('div.weather1_bodyUnitData')
                
                for item, value in zip(weather_items, weather_values):
                    key = item.text.strip()
                    val = value.text.strip()
                    weather_info[key] = val
            
            # 選手情報を取得
            racer_info = []
            racer_table = soup.select_one('table.is-w1265')
            
            if racer_table:
                rows = racer_table.select('tbody tr')
                
                for row in rows:
                    cells = row.select('td')
                    
                    if len(cells) >= 10:
                        rank = cells[0].text.strip()
                        waku = int(cells[1].text.strip())
                        racer_no = cells[2].text.strip()
                        racer_name = cells[3].text.strip()
                        course = int(cells[4].text.strip())
                        st_time = cells[5].text.strip()
                        race_time = cells[6].text.strip()
                        
                        racer_info.append({
                            'rank': rank,
                            'waku': waku,
                            'racer_no': racer_no,
                            'racer_name': racer_name,
                            'course': course,
                            'st_time': st_time,
                            'race_time': race_time
                        })
            
            result = {
                'date': date_str,
                'race_number': race_num,
                'weather_info': weather_info,
                'racer_info': racer_info
            }
            
            logger.info(f"Successfully fetched race details for date: {date_str}, race: {race_num}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching race details for date {date_str}, race {race_num}: {e}")
            return None
    
    def get_racer_info(self, racer_no):
        """
        選手情報を取得する
        
        Parameters:
        -----------
        racer_no : str
            選手登録番号
            
        Returns:
        --------
        dict
            選手情報
        """
        url = f"https://www.boatrace.jp/owpc/pc/data/racersearch/profile?toban={racer_no}"
        logger.info(f"Fetching racer info for racer number: {racer_no}, URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 選手基本情報を取得
            profile_table = soup.select_one('div.is-p10')
            
            if not profile_table:
                logger.warning(f"No profile information found for racer: {racer_no}")
                return None
            
            # 選手名を取得
            name_elem = soup.select_one('div.is-first')
            racer_name = name_elem.text.strip() if name_elem else "Unknown"
            
            # 基本情報を取得
            info_items = profile_table.select('dl')
            profile_info = {}
            
            for item in info_items:
                key_elem = item.select_one('dt')
                val_elem = item.select_one('dd')
                
                if key_elem and val_elem:
                    key = key_elem.text.strip()
                    val = val_elem.text.strip()
                    profile_info[key] = val
            
            # 成績情報を取得
            performance_table = soup.select_one('table.is-w238')
            performance_info = {}
            
            if performance_table:
                rows = performance_table.select('tr')
                
                for row in rows:
                    cells = row.select('td')
                    
                    if len(cells) >= 2:
                        key = row.select_one('th').text.strip() if row.select_one('th') else "Unknown"
                        val = cells[0].text.strip()
                        performance_info[key] = val
            
            result = {
                'racer_no': racer_no,
                'racer_name': racer_name,
                'profile_info': profile_info,
                'performance_info': performance_info
            }
            
            logger.info(f"Successfully fetched racer info for racer number: {racer_no}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching racer info for racer number {racer_no}: {e}")
            return None
    
    def get_motor_boat_info(self, date_str):
        """
        モーター・ボート情報を取得する
        
        Parameters:
        -----------
        date_str : str
            日付（YYYYMMDD形式）
            
        Returns:
        --------
        dict
            モーター・ボート情報
        """
        url = f"{self.base_url}motor?jcd={self.toda_jcd}&hd={date_str}"
        logger.info(f"Fetching motor/boat info for date: {date_str}, URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # モーター情報テーブルを取得
            motor_tables = soup.select('table.is-w844')
            
            if not motor_tables:
                logger.warning(f"No motor/boat information found for date: {date_str}")
                return None
            
            motor_boat_info = []
            
            for table in motor_tables:
                rows = table.select('tbody tr')
                
                for row in rows:
                    cells = row.select('td')
                    
                    if len(cells) >= 6:
                        motor_no = cells[0].text.strip()
                        boat_no = cells[1].text.strip()
                        motor_2rate = cells[2].text.strip()
                        motor_3rate = cells[3].text.strip()
                        boat_2rate = cells[4].text.strip()
                        boat_3rate = cells[5].text.strip()
                        
                        motor_boat_info.append({
                            'motor_no': motor_no,
                            'boat_no': boat_no,
                            'motor_2rate': motor_2rate,
                            'motor_3rate': motor_3rate,
                            'boat_2rate': boat_2rate,
                            'boat_3rate': boat_3rate
                        })
            
            result = {
                'date': date_str,
                'motor_boat_info': motor_boat_info
            }
            
            logger.info(f"Successfully fetched motor/boat info for date: {date_str}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching motor/boat info for date {date_str}: {e}")
            return None
    
    def collect_data_for_period(self, start_date, end_date):
        """
        指定した期間のデータを収集する
        
        Parameters:
        -----------
        start_date : str
            開始日（YYYYMMDD形式）
        end_date : str
            終了日（YYYYMMDD形式）
            
        Returns:
        --------
        dict
            収集したデータ
        """
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        logger.info(f"Starting data collection for period: {start_date} to {end_date}")
        
        all_results = []
        all_details = []
        all_motor_boat_info = []
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y%m%d')
            logger.info(f"Collecting data for {date_str}...")
            
            # レース結果を取得
            results = self.get_race_results(date_str)
            
            if results is not None and not results.empty:
                all_results.append(results)
                logger.info(f"Collected {len(results)} race results for {date_str}")
                
                # 各レースの詳細情報を取得
                for race_num in results['race_number'].unique():
                    details = self.get_race_details(date_str, race_num)
                    
                    if details is not None:
                        all_details.append(details)
                        logger.info(f"Collected details for race {race_num} on {date_str}")
                    
                    # サーバーに負荷をかけないよう少し待機
                    time.sleep(1)
            else:
                logger.warning(f"No races found for {date_str}")
            
            # モーター・ボート情報を取得
            motor_boat_info = self.get_motor_boat_info(date_str)
            
            if motor_boat_info is not None:
                all_motor_boat_info.append(motor_boat_info)
                logger.info(f"Collected motor/boat info for {date_str}")
            
            # サーバーに負荷をかけないよう少し待機
            time.sleep(2)
            
            current_date += timedelta(days=1)
        
        # 収集したデータを保存
        collected_data = {
            'period': {
                'start_date': start_date,
                'end_date': end_date
            },
            'race_results': all_results,
            'race_details': all_details,
            'motor_boat_info': all_motor_boat_info
        }
        
        # データをファイルに保存
        self._save_collected_data(collected_data)
        
        logger.info(f"Data collection completed for period: {start_date} to {end_date}")
        
        return collected_data
    
    def _save_collected_data(self, collected_data):
        """
        収集したデータをファイルに保存する
        
        Parameters:
        -----------
        collected_data : dict
            収集したデータ
        """
        start_date = collected_data['period']['start_date']
        end_date = collected_data['period']['end_date']
        
        # レース結果をCSVに保存
        if collected_data['race_results']:
            combined_results = pd.concat(collected_data['race_results'], ignore_index=True)
            results_file = os.path.join(self.processed_data_dir, f"toda_race_results_{start_date}_to_{end_date}.csv")
            combined_results.to_csv(results_file, index=False)
            logger.info(f"Race results saved to {results_file}")
        
        # レース詳細情報をJSONに保存
        if collected_data['race_details']:
            details_file = os.path.join(self.raw_data_dir, f"toda_race_details_{start_date}_to_{end_date}.json")
            with open(details_file, 'w', encoding='utf-8') as f:
                json.dump(collected_data['race_details'], f, ensure_ascii=False, indent=4)
            logger.info(f"Race details saved to {details_file}")
        
        # モーター・ボート情報をJSONに保存
        if collected_data['motor_boat_info']:
            motor_boat_file = os.path.join(self.raw_data_dir, f"toda_motor_boat_info_{start_date}_to_{end_date}.json")
            with open(motor_boat_file, 'w', encoding='utf-8') as f:
                json.dump(collected_data['motor_boat_info'], f, ensure_ascii=False, indent=4)
            logger.info(f"Motor/boat info saved to {motor_boat_file}")

# 使用例
if __name__ == "__main__":
    # ログディレクトリを作成
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs'), exist_ok=True)
    
    collector = BoatraceDataCollector()
    
    # 過去7日間のデータを収集する例
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    collector.collect_data_for_period(start_date_str, end_date_str)
