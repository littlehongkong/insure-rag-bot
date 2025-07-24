import requests
import supabase
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time
import re
import json
import glob

load_dotenv()  # .env 파일 로드

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DisclosureCollector:
    def __init__(self, raw_data_dir: str = "raw_data", processed_data_dir: str = "processed_data"):
        self.base_url = "https://kpub.knia.or.kr"
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)

        # 디렉토리 설정
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self._ensure_directories()
        self.table_info = {
            'PB': {
                'table_name': 'insurance_products_raw',
                'on_conflict': "TP_CODE,TP_PAY_NAME,popup_code,ROW_NUM"
            },
            'PC': {
                'table_name': 'insurance_products_long_term_savings_raw',
                'on_conflict': "TP_CODE,TP_PAY_NAME,popup_code,PERIOD,NUM1"
            }
        }

        # 오늘 날짜
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.date_str = datetime.now().strftime('%Y%m%d')

        self.popup = {
            1: {
                'menu_name': '장기보장성',
                'data': {
                    "화재": "PB11",
                    "종합": "PB12",
                    "운전자": "PB13",
                    "어린이": "PB24",
                    "치아": "PB25",
                    "간병·치매": "PB16",
                    "암": "PB22",
                    "상해": "PB14",
                    "질병": "PB15",
                    "비용": "PB18",
                    "기타": "PB17"
                }
            },
            2: {
                'menu_name': '장기저축성',
                'data': {
                    "장기화재 및 종합보험": "PC11",
                    "장기상해보험 및 기타": "PC12",
                }
            },
            3: {
                'menu_name': '실손의료보험',
                'data': {
                    "실손의료보험(4세대)": "PB26",
                    "노후실손의료보험": "PB21",
                    "유병력자실손보험": "PB23",
                }
            },
            4: {
                'menu_name': '연금저축',
                'data': {
                    "연금저축": "PC99"
                }
            }
        }

    def _ensure_directories(self):
        """필요한 디렉토리들을 생성합니다."""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        logging.info(f"✅ 디렉토리 준비 완료: {self.raw_data_dir}, {self.processed_data_dir}")

    def save_raw_response(self, response_text: str, popup_code: str, category_main: str, category_sub: str) -> str:
        """원천 API 응답을 파일로 저장합니다."""
        try:
            # 파일명에 카테고리 정보 포함
            safe_category_sub = re.sub(r'[^\w\-_]', '_', category_sub)  # 파일명에 사용할 수 없는 문자 제거
            filename = os.path.join(
                self.raw_data_dir,
                f"{self.date_str}_{popup_code}_{safe_category_sub}_raw.json"
            )

            # 메타데이터와 함께 저장
            raw_data = {
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "popup_code": popup_code,
                "category_main": category_main,
                "category_sub": category_sub,
                "raw_response": response_text
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            logging.info(f"✅ 원천 데이터 저장: {filename}")
            return filename

        except Exception as e:
            logging.error(f"❌ 원천 데이터 저장 실패: {e}")
            raise

    def collect_raw_data(self) -> bool:
        """모든 카테고리의 원천 데이터를 수집하여 로컬에 저장합니다."""
        url = f"{self.base_url}/popup/disclosureList.do"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'https://kpub.knia.or.kr',
            'Referer': 'https://kpub.knia.or.kr/productDisc/longTermGuarantee/fireInsurance.do'
        }

        total_categories = sum(len(self.popup[key]['data']) for key in self.popup.keys())
        current_count = 0
        success_count = 0

        logging.info(f"🚀 원천 데이터 수집 시작 - 총 {total_categories}개 카테고리")

        for tab_key in self.popup.keys():
            category_main = self.popup[tab_key]['menu_name']

            for category_sub, popup_code in self.popup[tab_key]['data'].items():
                current_count += 1
                logging.info(f"📥 수집 중 ({current_count}/{total_categories}): {category_main} > {category_sub}")

                payload = {
                    'tabType': str(tab_key),
                    'tptyCode': popup_code,
                    'refreshYn': '',
                    'detailYn': '',
                    'pCode': '',
                    'channel': '',
                    'prdNm': '',
                    'payNm': '',
                    'payReason': '',
                    '__sort__': '1 asc'
                }

                try:
                    # SSL 검증과 함께 요청 시도
                    try:
                        response = requests.post(url, data=payload, headers=headers, verify=True, timeout=30)
                        response.raise_for_status()
                    except requests.exceptions.SSLError as ssl_err:
                        logging.warning(f"SSL 검증 실패, SSL 없이 재시도: {ssl_err}")
                        import urllib3
                        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                        response = requests.post(url, data=payload, headers=headers, verify=False, timeout=30)
                        response.raise_for_status()

                    # 원천 데이터 저장
                    saved_file = self.save_raw_response(
                        response.text, popup_code, category_main, category_sub
                    )

                    if saved_file:
                        success_count += 1
                        logging.info(f"✅ 저장 완료 ({success_count}/{current_count})")
                    else:
                        logging.error(f"❌ 저장 실패: {category_sub}")

                    # 서버 부하 방지를 위한 대기
                    time.sleep(1)

                except requests.exceptions.RequestException as e:
                    logging.error(f"❌ API 요청 실패 ({category_sub}): {e}")
                except Exception as e:
                    logging.error(f"❌ 예상치 못한 오류 ({category_sub}): {e}")

        logging.info(f"🎉 원천 데이터 수집 완료: {success_count}/{total_categories}개 성공")
        return success_count > 0

    def parse_api_response(self, api_response_string: str) -> dict:
        """
        API 응답을 파싱하고 유효성을 검사합니다.
        불완전하거나 잘못된 JSON 형식을 보정하여 파싱을 시도합니다.
        """
        # 1. 문자열 앞뒤 공백 제거
        cleaned_string = api_response_string.strip()

        # 2. HTML 태그 (<i class="noViewItem">...</i> 패턴) 제거
        cleaned_string = re.sub(r'<i[^>]*?>.*?</i>', '', cleaned_string, flags=re.DOTALL)

        # 3. 이스케이프되지 않은 '\r', '\n', '\t'을 JSON 표준에 맞게 '\\r', '\\n', '\\t'으로 변경
        #    이 단계는 유지합니다.
        corrected_string = cleaned_string.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')

        # 4. 키에 따옴표가 없는 경우 큰따옴표로 감싸기
        corrected_string = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', corrected_string)

        # 5. 작은따옴표로 묶인 문자열 값을 큰따옴표로 변경
        corrected_string = re.sub(r"'([^']*)'", r'"\1"', corrected_string)

        # ✨ 6. 'TP_ETC' 필드의 값에서 발생하는 잘못된 따옴표 및 이스케이프된 문자열 처리
        #    "TP_ETC":""값"" 형태 처리 (이전 단계에서 추가된 로직)
        corrected_string = re.sub(r'"TP_ETC":\s*""(.*?)"",', r'"TP_ETC":"\1",', corrected_string)

        try:
            # 1차 파싱 시도 (가장 깔끔한 상태)
            data = json.loads(corrected_string)
            return data
        except json.JSONDecodeError as e:
            logging.warning(f"1차 정리 후 JSON 파싱 오류: {e}. 추가 보정 시도.")
            # 2차 보정: 작은 따옴표를 큰 따옴표로 변경 (키에 대한 처리는 위에서 이미 시도됨)
            temp_string = cleaned_string.replace("'", '"')
            try:
                data = json.loads(temp_string)
                return data
            except json.JSONDecodeError as inner_e:
                logging.warning(f"2차 정리 후에도 파싱 실패: {inner_e}")

        # 마지막 시도: Regex를 통한 최종 보정
        try:
            # JSON 배열의 시작과 끝이 불분명할 때 전체를 [ ]로 감싸기
            if not cleaned_string.startswith('[') and not cleaned_string.endswith(']'):
                # 만약 "list": [ ... ] 형태의 최상위 객체가 아니라,
                # 바로 배열 내용만 넘어오는 경우를 대비 (드물지만 발생 가능)
                if cleaned_string.startswith('{') and cleaned_string.endswith('}'):
                    # 단일 객체일 경우 배열로 감싸기 (ex: {"a":1} -> [{"a":1}])
                    temp_string = f"[{cleaned_string}]"
                elif 'list' in cleaned_string and not cleaned_string.strip().startswith('{'):
                    # "list": [...] 형태인데 시작이 { 가 아닌 경우 (잘못된 API)
                    # 이건 매우 복잡하므로, 일단은 최상위 객체 형태만 고려
                    pass

            # JSON 내의 잘못된 문자열 이스케이프를 제거하거나 올바르게 변환.
            # 예: "abc\"def" (제대로 이스케이프 안 된 따옴표) -> "abc\\"def" (더블 이스케이프) 또는 "abc def" (제거)
            # 여기서는 가장 흔한 패턴인 문자열 내의 이스케이프되지 않은 따옴표를 `\"`로 바꾸는 시도
            # (이것은 매우 주의해야 함: `re.sub(r'(?<!\\)"', '\"', temp_string)`와 같이 하면 기존 `\"`까지 `\\\"`로 바꿀 수 있음)
            # 가장 안전한 방법은 `json.dumps(value)`를 사용하는 것이지만, 이는 이미 파싱된 객체에만 가능.
            # 그래서 아래와 같이 패턴을 사용하되, 특정 키에만 적용하는 것이 더 정확할 수 있습니다.
            # 예: "TP_ETC":"Some text with "unclosed quote""
            # 이 패턴은 매우 까다로우므로, 다음과 같은 시도를 해볼 수 있습니다.
            # 1. 필드 값이 HTML 태그를 포함하는 경우: `<[^>]*>` 제거
            # 2. 필드 값이 이스케이프되지 않은 따옴표를 포함하는 경우: 해당 따옴표 앞에 백슬래시 추가

            # JSON 문자열 내의 이스케이프되지 않은 " (따옴표)를 찾아서 이스케이프 (매우 위험한 시도일 수 있음)
            # cleaned_string = re.sub(r'(?<!\\)"(?![,:}\]])', r'\"', cleaned_string)
            # 위 패턴은 오류를 유발할 가능성이 높음. 대신, JSON 파서가 허용하지 않는 문자열 내의 제어 문자를 제거.
            # (이 부분은 위에서 [\x00-\x1f] 처리로 이미 시도됨)

            # 다시 json.loads 시도
            data = json.loads(cleaned_string)  # 최종적으로 보정된 문자열로 파싱 시도
            return data

        except json.JSONDecodeError as final_e:
            logging.error(f"❌ 최종 JSON 파싱 실패: {final_e}. 원본 문자열의 일부를 출력합니다: {cleaned_string[:500]}...")
            return {
                "list": [],
                "totalCount": 0,
                "result": "ERROR",
                "resultMsg": f"최종 파싱 오류: {str(final_e)}"
            }
        except Exception as unknown_e:
            logging.error(f"❌ 예기치 않은 파싱 오류: {unknown_e}")
            return {
                "list": [],
                "totalCount": 0,
                "result": "ERROR",
                "resultMsg": f"예기치 않은 파싱 오류: {str(unknown_e)}"
            }

    def process_raw_data(self) -> List[Dict]:
        """저장된 원천 데이터를 모두 처리하여 정제된 데이터를 반환합니다."""
        raw_files = glob.glob(os.path.join(self.raw_data_dir, f"{self.date_str}_*_raw.json"))

        if not raw_files:
            logging.warning(f"⚠️ 처리할 원천 데이터 파일이 없습니다: {self.raw_data_dir}")
            return []

        all_processed_data = []
        successful_files = 0

        logging.info(f"🔄 데이터 처리 시작: {len(raw_files)}개 파일")

        for file_path in raw_files:
            try:
                logging.info(f"📝 처리 중: {os.path.basename(file_path)}")

                # 원천 데이터 로드
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                # API 응답 파싱
                parsed_data = self.parse_api_response(raw_data['raw_response'])

                # 데이터 유효성 검사 및 처리
                if 'list' in parsed_data and isinstance(parsed_data['list'], list):
                    items = parsed_data['list']

                    # 각 아이템에 메타데이터 추가
                    for item in items:
                        item.update({
                            'popup_category_main': raw_data['category_main'],
                            'popup_category_sub': raw_data['category_sub'],
                            'popup_code': raw_data['popup_code']
                        })

                    all_processed_data.extend(items)
                    successful_files += 1

                    logging.info(f"✅ 처리 완료: {len(items)}개 아이템 추가")
                else:
                    logging.warning(f"⚠️ 유효하지 않은 데이터 구조: {os.path.basename(file_path)}")

            except Exception as e:
                logging.error(f"❌ 파일 처리 실패 ({os.path.basename(file_path)}): {e}")

        logging.info(f"🎉 데이터 처리 완료: {successful_files}/{len(raw_files)}개 파일, 총 {len(all_processed_data)}개 아이템")
        return all_processed_data

    def save_processed_data(self, processed_data: List[Dict]) -> bool:
        """처리된 데이터를 카테고리별로 JSON 파일에 저장합니다."""
        if not processed_data:
            logging.info("저장할 처리된 데이터가 없습니다.")
            return False

        try:
            # 카테고리별로 데이터 그룹화
            grouped_data = {}
            for item in processed_data:
                popup_code = item.get('popup_code', 'unknown')
                if popup_code not in grouped_data:
                    grouped_data[popup_code] = []
                grouped_data[popup_code].append(item)

            # 각 그룹을 별도 파일로 저장
            for popup_code, items in grouped_data.items():
                filename = os.path.join(self.processed_data_dir, f"{self.date_str}_{popup_code}_processed.json")

                processed_file_data = {
                    'popup_code': popup_code,
                    'processed_at': datetime.utcnow().isoformat() + 'Z',
                    'count': len(items),
                    'data': items
                }

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(processed_file_data, f, ensure_ascii=False, indent=2)

                logging.info(f"✅ 처리된 데이터 저장: {filename} ({len(items)}개 아이템)")

            return True

        except Exception as e:
            logging.error(f"❌ 처리된 데이터 저장 실패: {e}")
            return False

    def _remove_full_duplicates(self, data_list: List[Dict]) -> List[Dict]:
        """
        데이터 리스트에서 모든 필드의 값이 동일한 완전 중복 레코드를 제거합니다.
        """
        seen_records = set()
        unique_data = []
        original_count = len(data_list)

        for record in data_list:
            # 딕셔너리를 JSON 문자열로 변환하여 해시 가능하게 만듭니다.
            # 키 순서를 정렬하여 같은 내용이라도 순서가 다르면 다른 것으로 인식하지 않도록 합니다.
            try:
                record_str = json.dumps(record, sort_keys=True, ensure_ascii=False)
            except TypeError as e:
                logging.warning(
                    f"⚠️ 레코드 직렬화 실패 (중복 제거 건너뛰기): {e} - {record.get('TP_NAME', 'Unknown Product')}. 이 레코드는 중복 제거 대상에서 제외됩니다.")
                # 직렬화 실패 시, 해당 레코드는 고유한 것으로 간주하고 추가합니다.
                unique_data.append(record)
                continue

            if record_str not in seen_records:
                seen_records.add(record_str)
                unique_data.append(record)
            else:
                logging.info(f"🧹 완전 중복 레코드 제거: {record.get('TP_CODE', 'N/A')}, {record.get('TP_PAY_NAME', 'N/A')}")

        logging.info(f"✅ 전체 중복 제거 완료: {original_count}개 레코드 중 {len(unique_data)}개 유니크")
        return unique_data

    def store_in_supabase(self, processed_data: List[Dict]) -> bool:
        """처리된 데이터를 Supabase에 저장합니다. 카테고리별로 다른 테이블과 제약 조건을 사용합니다."""
        if not processed_data:
            logging.info("DB에 저장할 데이터가 없습니다.")
            return False

        # Supabase 저장 전, 전체 데이터에 대해 완전 중복 제거 수행
        logging.info(f"✨ Supabase 저장 전, 전체 {len(processed_data)}개 레코드에서 완전 중복 제거 시작...")
        processed_data = self._remove_full_duplicates(processed_data)
        logging.info(f"✨ 완전 중복 제거 후 {len(processed_data)}개 레코드 저장 시도")

        total_overall_success = True
        current_time = datetime.utcnow().isoformat() + 'Z'

        # 데이터를 table_info에 정의된 카테고리별로 그룹화
        grouped_by_table_type = {}
        for item in processed_data:
            # popup_code의 앞 두글자를 사용하여 테이블 타입을 결정
            popup_code_prefix = item.get('popup_code', '')[:2]
            if popup_code_prefix in self.table_info:
                if popup_code_prefix not in grouped_by_table_type:
                    grouped_by_table_type[popup_code_prefix] = []
                grouped_by_table_type[popup_code_prefix].append(item)
            else:
                logging.warning(
                    f"⚠️ 알 수 없는 popup_code 접두사 '{popup_code_prefix}'. 이 데이터는 저장되지 않습니다: {item.get('TP_NAME')}")
                total_overall_success = False

        for table_type, items_for_table in grouped_by_table_type.items():
            table_name = self.table_info[table_type]['table_name']
            on_conflict_cols_str = self.table_info[table_type]['on_conflict']
            on_conflict_cols_list = [col.strip() for col in on_conflict_cols_str.split(',')]

            logging.info(f"💾 '{table_name}' 테이블에 저장 시작: 총 {len(items_for_table)}개 아이템")

            batch_size = 100
            table_success_count = 0

            # 배치를 나눠서 처리
            for i in range(0, len(items_for_table), batch_size):
                batch = items_for_table[i:i + batch_size]

                # 각 레코드에 updated_at 추가
                # on_conflict_cols를 기준으로 한 중복 제거는 _remove_full_duplicates가 처리했으므로,
                # 여기서는 on_conflict 컬럼의 None 값 처리만 신경쓰면 됩니다.
                # 그러나 _remove_full_duplicates가 모든 필드를 JSON화하므로,
                # 여기서는 on_conflict 필드 값에 None이 있는지 다시 검사할 필요는 크게 없습니다.
                # (만약 on_conflict 필드 자체가 JSON 직렬화에 영향을 주지 않는다면).

                # 실제 Supabase UPSERT를 위한 데이터 준비
                final_batch_for_supabase = []
                for item in batch:
                    # on_conflict 컬럼에 None이 있으면 Supabase에서 오류 발생 가능성
                    # 따라서 이 부분에서 다시 한번 확인하고, 필요시 대체 값 적용
                    valid_item = True
                    for col in on_conflict_cols_list:
                        if item.get(col) is None:
                            logging.warning(
                                f"❌ '{table_name}' 테이블의 on_conflict 컬럼 '{col}'에 None 값 발견. "
                                f"레코드 스킵 또는 기본값으로 대체: {item.get('TP_NAME', 'Unknown')}"
                            )
                            # None 값을 포함한 레코드를 건너뛸지, 빈 문자열 등으로 대체할지 결정
                            # 여기서는 일단 스킵하지 않고, Supabase가 None을 어떻게 처리하는지 확인합니다.
                            # Supabase는 ON CONFLICT에서 NULL 값을 허용하지 않을 수 있습니다.
                            # 만약 오류가 발생한다면, 이 부분에서 None 값을 "" 등으로 대체하는 로직 추가 필요.
                            # 예: item[col] = "" if item.get(col) is None else item[col]
                            pass  # 현재는 스킵하지 않고 그대로 전달

                    final_batch_for_supabase.append({**item, 'updated_at': current_time})

                if not final_batch_for_supabase:
                    logging.info(f"✅ 배치 {i // batch_size + 1}에 Supabase에 저장할 유효한 항목이 없습니다. 건너뛰기.")
                    continue

                try:
                    # Bulk UPSERT 실행
                    response = self.supabase.table(table_name).upsert(
                        final_batch_for_supabase,
                        on_conflict=on_conflict_cols_str  # 동적으로 on_conflict 설정
                    ).execute()

                    batch_success = len(response.data) if response.data else 0
                    table_success_count += batch_success

                    logging.info(
                        f"✅ '{table_name}' 배치 저장 ({i // batch_size + 1}/{(len(items_for_table) + batch_size - 1) // batch_size}): "
                        f"{batch_success}/{len(final_batch_for_supabase)}개 성공"
                    )

                except Exception as e:
                    logging.error(f"❌ '{table_name}' 배치 저장 실패 (배치 {i // batch_size + 1}): {e}")
                    total_overall_success = False

                    # 실패 시 개별 레코드로 재시도 (중복 제거된 배치에 대해서만)
                    logging.info("개별 레코드로 재시도 중...")
                    for record in final_batch_for_supabase:  # deduplicated_batch 대신 final_batch_for_supabase 사용
                        try:
                            individual_response = self.supabase.table(table_name).upsert(
                                {**record},  # 이미 updated_at 추가되어 있으므로 다시 추가하지 않음
                                on_conflict=on_conflict_cols_str  # 동적으로 on_conflict 설정
                            ).execute()

                            if individual_response.data:
                                table_success_count += 1
                                logging.info(f"✅ 개별 저장 성공: {record.get('TP_NAME', 'Unknown')}")

                        except Exception as individual_e:
                            logging.error(f"❌ 개별 저장 실패: {record.get('TP_NAME', 'Unknown')} - {individual_e}")
                            total_overall_success = False

            logging.info(f"🎉 '{table_name}' DB 저장 완료: {table_success_count}/{len(items_for_table)}개 성공")

        if not grouped_by_table_type:
            logging.warning("⚠️ 저장할 테이블 유형이 정의되지 않은 데이터가 많거나, 처리된 데이터가 없습니다.")
            return False

        return total_overall_success

    def cleanup_old_files(self, days_to_keep: int = 7):
        """지정된 일수보다 오래된 파일들을 정리합니다."""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)

            for directory in [self.raw_data_dir, self.processed_data_dir]:
                files = glob.glob(os.path.join(directory, "*.json"))
                deleted_count = 0

                for file_path in files:
                    if os.path.getmtime(file_path) < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1

                if deleted_count > 0:
                    logging.info(f"🧹 정리 완료 ({directory}): {deleted_count}개 파일 삭제")

        except Exception as e:
            logging.error(f"❌ 파일 정리 실패: {e}")

    def run_full_pipeline(self, cleanup_days: int = 7):
        """전체 파이프라인을 실행합니다: 수집 → 처리 → 저장 → 정리"""
        pipeline_start_time = datetime.now() - timedelta(days=1)
        # 오늘 날짜
        self.today = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        self.date_str = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

        logging.info(f"🚀 전체 파이프라인 시작: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # # 1. 원천 데이터 수집
            # logging.info("=" * 50)
            # logging.info("1단계: 원천 데이터 수집")
            # logging.info("=" * 50)
            #
            # if not self.collect_raw_data():
            #     logging.error("❌ 원천 데이터 수집 실패")
            #     return False

            # # 2. 데이터 처리
            # logging.info("=" * 50)
            # logging.info("2단계: 데이터 처리 및 보정")
            # logging.info("=" * 50)
            #
            processed_data = self.process_raw_data()
            if not processed_data:
                logging.error("❌ 데이터 처리 실패")
                return False

            # 3. 처리된 데이터 로컬 저장
            logging.info("=" * 50)
            logging.info("3단계: 처리된 데이터 로컬 저장")
            logging.info("=" * 50)

            if not self.save_processed_data(processed_data):
                logging.warning("⚠️ 처리된 데이터 로컬 저장 실패 (계속 진행)")

            # 4. DB 저장
            logging.info("=" * 50)
            logging.info("4단계: 데이터베이스 저장")
            logging.info("=" * 50)

            if not self.store_in_supabase(processed_data):
                logging.error("❌ DB 저장 실패")
                return False

            # 5. 오래된 파일 정리
            if cleanup_days > 0:
                logging.info("=" * 50)
                logging.info("5단계: 파일 정리")
                logging.info("=" * 50)
                self.cleanup_old_files(cleanup_days)

            # 완료
            pipeline_end_time = datetime.now()
            duration = pipeline_end_time - pipeline_start_time
            logging.info("=" * 50)
            logging.info(f"🎉 전체 파이프라인 완료!")
            logging.info(f"   소요 시간: {duration}")
            logging.info(f"   처리된 데이터: {len(processed_data)}개")
            logging.info("=" * 50)

            return True

        except Exception as e:
            logging.error(f"❌ 파이프라인 실행 중 오류: {e}")
            import traceback
            logging.error(f"오류 상세: {traceback.format_exc()}")
            return False

    # 기존 메서드들과의 호환성을 위한 래퍼 메서드들
    def collect_daily_disclosures(self):
        """기존 호환성을 위한 메서드 (새로운 파이프라인 사용)"""
        return self.run_full_pipeline()

    def run(self):
        """기존 호환성을 위한 메서드 (새로운 파이프라인 사용)"""
        return self.run_full_pipeline()


if __name__ == "__main__":
    # 사용 예시
    collector = DisclosureCollector(
        raw_data_dir="raw_data",  # 원천 데이터 저장 디렉토리
        processed_data_dir="processed_data"  # 처리된 데이터 저장 디렉토리
    )

    # 전체 파이프라인 실행
    success = collector.run_full_pipeline(cleanup_days=7)

    if success:
        print("✅ 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("❌ 작업 중 오류가 발생했습니다. 로그를 확인해주세요.")