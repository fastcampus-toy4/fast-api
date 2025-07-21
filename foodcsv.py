import requests
import xml.etree.ElementTree as ET
import csv
import math
import os
from dotenv import load_dotenv

load_dotenv()
encoded_key = os.getenv("ENCODED_KEY")

# 기본 URL
base_url = 'http://api.data.go.kr/openapi/tn_pubr_public_nutri_food_info_api'
num_of_rows = 100

# 전체 데이터 개수 먼저 요청해서 totalCount 구하기
first_url = f'{base_url}?serviceKey={encoded_key}&pageNo=1&numOfRows=1&type=xml'
first_response = requests.get(first_url)
first_root = ET.fromstring(first_response.content)

# totalCount 추출
total_count = int(first_root.find('.//totalCount').text)
total_pages = math.ceil(total_count / num_of_rows)

print(f'총 {total_count}건, 총 {total_pages}페이지')

# CSV 저장
with open('nutri_food_info.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    headers_written = False

    for page in range(1, total_pages + 1):
        url = f'{base_url}?serviceKey={encoded_key}&pageNo={page}&numOfRows={num_of_rows}&type=xml'
        response = requests.get(url)
        root = ET.fromstring(response.content)

        for item in root.iter('item'):
            row = []
            headers = []

            for elem in item:
                headers.append(elem.tag)
                row.append(elem.text)

            if not headers_written:
                writer.writerow(headers)
                headers_written = True

            writer.writerow(row)

print("✅ 모든 페이지 크롤링 완료, CSV 저장됨.")
