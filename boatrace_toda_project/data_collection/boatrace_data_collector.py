import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime, timedelta

class BoatraceDataCollector:
    """
    ボートレース戸田のデータを収集するクラス
    """
    
    def __init__(self):
        self.base_url = "https://www.boatrace.jp/owpc/pc/race/"
        self.toda_jcd = "02"  # 戸田競艇場のコード
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(self.data_dir, exist_ok=True)
    
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
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # レース結果テーブルを取得
            result_tables = soup.select('table.is-w495')
            
            if not result_tables:
                print(f"No race results found for date: {date_str}")
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
            
            return pd.DataFrame(all_results)
            
        except Exception as e:
            print(f"Error fetching race results for date {date_str}: {e}")
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
            
            return {
                'date': date_str,
                'race_number': race_num,
                'weather_info': weather_info,
                'racer_info': racer_info
            }
            
        except Exception as e:
            print(f"Error fetching race details for date {date_str}, race {race_num}: {e}")
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
        pandas.DataFrame
            収集したデータのデータフレーム
        """
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        all_results = []
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y%m%d')
            print(f"Collecting data for {date_str}...")
            
            results = self.get_race_results(date_str)
            
            if results is not None and not results.empty:
                all_results.append(results)
                print(f"Collected {len(results)} races for {date_str}")
            else:
                print(f"No races found for {date_str}")
            
            # サーバーに負荷をかけないよう少し待機
            time.sleep(1)
            
            current_date += timedelta(days=1)
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # CSVファイルに保存
            output_file = os.path.join(self.data_dir, f"toda_race_results_{start_date}_to_{end_date}.csv")
            combined_results.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
            
            return combined_results
        else:
            print("No data collected for the specified period")
            return None
    
    def analyze_course_statistics(self, df):
        """
        コース別の統計情報を分析する
        
        Parameters:
        -----------
        df : pandas.DataFrame
            レース結果のデータフレーム
            
        Returns:
        --------
        dict
            コース別統計情報
        """
        if df is None or df.empty:
            return None
        
        # 1着のコース別集計
        first_place_counts = df['first'].value_counts().sort_index()
        first_place_percentage = (first_place_counts / len(df) * 100).round(1)
        
        # 2着のコース別集計
        second_place_counts = df['second'].value_counts().sort_index()
        second_place_percentage = (second_place_counts / len(df) * 100).round(1)
        
        # 3着のコース別集計
        third_place_counts = df['third'].value_counts().sort_index()
        third_place_percentage = (third_place_counts / len(df) * 100).round(1)
        
        # 3連単の組み合わせ集計
        trifecta_combinations = df.apply(lambda row: f"{row['first']}-{row['second']}-{row['third']}", axis=1)
        top_combinations = trifecta_combinations.value_counts().head(10)
        
        return {
            'first_place': {
                'counts': first_place_counts.to_dict(),
                'percentage': first_place_percentage.to_dict()
            },
            'second_place': {
                'counts': second_place_counts.to_dict(),
                'percentage': second_place_percentage.to_dict()
            },
            'third_place': {
                'counts': third_place_counts.to_dict(),
                'percentage': third_place_percentage.to_dict()
            },
            'top_combinations': top_combinations.to_dict()
        }

# 使用例
if __name__ == "__main__":
    collector = BoatraceDataCollector()
    
    # 過去1ヶ月のデータを収集する例
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    results = collector.collect_data_for_period(start_date_str, end_date_str)
    
    if results is not None:
        # 統計情報を分析
        stats = collector.analyze_course_statistics(results)
        
        print("\n===== コース別1着率 =====")
        for course, percentage in stats['first_place']['percentage'].items():
            print(f"コース{course}: {percentage}%")
        
        print("\n===== 人気の3連単組み合わせ =====")
        for combo, count in stats['top_combinations'].items():
            print(f"{combo}: {count}回")
