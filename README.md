# 한국 주식 뉴스 기반 포트폴리오 대시보드

네이버 뉴스 검색 API, KIS Open API, pykrx를 조합해 한국 주식의 뉴스 흐름과 주가 변화를 함께 해석하는 Streamlit 기반 포트폴리오 프로젝트입니다.

이 프로젝트는 크게 두 가지 화면을 제공합니다.

- `app.py`
  실사용/개발용 전체 대시보드입니다. 뉴스 분석, 시장 스캔, 추천 후보, 모의투자 자동 운용, 포지션 모니터링 기능이 포함됩니다.
- `app_public.py`
  이력서·포트폴리오 공유용 읽기 전용 공개 화면입니다. 방문자는 현재 포지션, 최근 매매 이력, 추천 후보, 주가 그래프만 볼 수 있고 직접 조작할 수 없습니다.

## 주요 기능

- 한국 주식 종목 검색 및 종목 목록 선택
- 네이버 뉴스 기반 최근 기사 수집
- 간단한 감성 점수 및 이벤트 태그 분석
- 뉴스와 주가를 결합한 단기 영향도 해석
- 시장 전체 후보 스캔과 상승 기대 종목 추천
- KIS 모의투자 기반 자동 매수/매도 흐름
- 포지션별 손익, 평가금액, 누적 수익률 모니터링
- 공개용 포트폴리오 페이지에서 읽기 전용 성과 공유

## 프로젝트 구조

```text
mini_project/
|- app.py
|- app_public.py
|- requirements.txt
|- .env.example
|- secrets.public.example.toml
|- README.md
|- analysis/
|- data/
|- docs/
|- models/
|- services/
|- tools/
|- utils/
`- workers/
```

## 설치 방법

1. 가상환경을 생성하고 활성화합니다.
2. 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

3. `.env.example`을 복사해 `.env`를 만든 뒤 필요한 키를 채웁니다.

```env
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
KIS_APP_KEY=your_kis_app_key
KIS_APP_SECRET=your_kis_app_secret
KIS_MOCK_APP_KEY=your_kis_mock_app_key
KIS_MOCK_APP_SECRET=your_kis_mock_app_secret
KIS_ACCOUNT_NO=12345678
KIS_ACCOUNT_PRODUCT_CODE=01
```

## 로컬 실행

### 전체 기능 대시보드 실행

```bash
streamlit run app.py
```

이 화면에서는 다음 기능을 사용할 수 있습니다.

- 종목 뉴스 영향 분석
- 시장 스캔 및 추천 후보 확인
- 모의투자 자동 운용
- 포지션/매매 이력/손익 대시보드

### 공개용 읽기 전용 앱 실행

```bash
streamlit run app_public.py
```

이 화면에서는 다음만 표시됩니다.

- 현재 보유 포지션
- 최근 자동 매매 이력
- 최근 추천 후보
- 대표 종목 주가 그래프
- 보유 포지션 클릭 시 해당 종목 주가 변동 그래프

방문자는 매수/매도/설정 변경을 할 수 없습니다.

## 데이터 흐름

1. 사용자가 종목을 선택합니다.
2. `services/stock_service.py`가 KIS Open API로 국내 주가를 조회합니다.
3. KIS 조회가 불안정하거나 제한될 경우 pykrx를 보조 데이터 소스로 사용합니다.
4. `services/news_service.py`가 네이버 뉴스 검색 API로 최근 기사를 가져옵니다.
5. `analysis/sentiment.py`, `analysis/event_tags.py`가 기사별 감성과 이벤트 태그를 계산합니다.
6. `analysis/features.py`가 날짜 단위 뉴스/가격 특징을 생성합니다.
7. `analysis/scoring.py`가 영향도 점수와 방향성을 계산합니다.
8. 대시보드가 추천 후보, 차트, 손익, 매매 흐름을 화면에 표시합니다.

## 모의투자 흐름

현재 프로젝트의 거래 기능은 모의투자 기준으로 구성되어 있습니다.

1. KIS 모의투자 앱키와 앱시크릿을 준비합니다.
2. `.env`에 아래 값을 입력합니다.
   - `KIS_MOCK_APP_KEY`
   - `KIS_MOCK_APP_SECRET`
   - `KIS_ACCOUNT_NO`
   - `KIS_ACCOUNT_PRODUCT_CODE`
3. `streamlit run app.py`로 전체 앱을 실행합니다.
4. 종목 분석 또는 시장 스캔 결과를 바탕으로 모의 자동 운용을 진행합니다.
5. 포지션은 `data/mock_trade_state.json`에 저장됩니다.
6. 전략 상태 및 주문 기록은 `data/strategy_state.json`에 저장됩니다.

기본 자동매도 아이디어는 `예상 상승률의 일정 비율`을 목표 수익률로 사용하는 방식입니다.

예시:

- 예상 상승률이 `8%`라면
- 목표 수익률을 `4%` 정도로 설정해 자동 매도 기준으로 삼을 수 있습니다.

## 공개 배포용 앱 설명

`app_public.py`는 Streamlit Community Cloud 배포를 위한 읽기 전용 포트폴리오 앱입니다.

특징:

- 방문자는 포트폴리오 진행 상황만 볼 수 있습니다.
- 자동매매 설정, 매수/매도 버튼, 수동 조작 기능은 노출하지 않습니다.
- 공개 화면에서도 서버 쪽에서 모의 자동 운용을 이어갈 수 있습니다.
- 보유 포지션 표에서 종목을 클릭하면 아래에 해당 종목의 주가 그래프가 표시됩니다.

## 공개 배포용 Secrets 예시

공개 배포 시에는 [secrets.public.example.toml](./secrets.public.example.toml)을 참고해서 Streamlit Community Cloud Secrets에 입력하면 됩니다.

예시:

```toml
NAVER_CLIENT_ID = "your_naver_client_id"
NAVER_CLIENT_SECRET = "your_naver_client_secret"
KIS_APP_KEY = "your_kis_app_key"
KIS_APP_SECRET = "your_kis_app_secret"
KIS_MOCK_APP_KEY = "your_kis_mock_app_key"
KIS_MOCK_APP_SECRET = "your_kis_mock_app_secret"
KIS_ACCOUNT_NO = "12345678"
KIS_ACCOUNT_PRODUCT_CODE = "01"

PUBLIC_APP_ENABLE_AUTO_CYCLE = "false"
PUBLIC_APP_REFRESH_SECONDS = "15"
PUBLIC_APP_MIN_CYCLE_INTERVAL_SECONDS = "30"
PUBLIC_APP_CYCLE_ONCE_PER_DAY = "true"
PUBLIC_APP_PREFER_SEED_DATA = "true"
PUBLIC_APP_SPOTLIGHT_SYMBOL = "005930"
```

## Streamlit Community Cloud 배포 시 주의사항

- GitHub에는 `.env`를 올리지 않습니다.
- `.streamlit/secrets.toml`도 저장소에 올리지 않습니다.
- 공개 배포에서는 실전 계좌 키 대신 모의투자 키만 사용하는 것을 권장합니다.
- Streamlit Community Cloud의 `Main file path`는 반드시 `app_public.py`로 지정합니다.
- `PUBLIC_APP_PREFER_SEED_DATA=true`로 두면 공개 앱은 샘플 포트폴리오를 우선 표시합니다.
- `PUBLIC_APP_SPOTLIGHT_SYMBOL`을 지정하면 공개 화면 상단 대표 그래프 종목을 고정할 수 있습니다.

## GitHub 업로드 전 체크

- `.env`가 `.gitignore`에 포함되어 있는지 확인
- `data/*.json` 상태 파일이 Git에 포함되지 않는지 확인
- 공개용 앱은 `app_public.py`인지 확인
- 공개 저장소에는 민감한 키, 계좌 정보, 실전 투자용 설정을 올리지 않기

## 앞으로 확장할 수 있는 방향

- 뉴스 본문 요약 및 핵심 문장 추출
- 종목별 비교 화면 강화
- 예측 모델 고도화
- 섹터별 추천 후보 묶음 제공
- 포트폴리오 성과 리포트 자동 생성

## 참고

- 전체 기능 개발/운영: `app.py`
- 공개용 읽기 전용 포트폴리오: `app_public.py`
- 공개 배포용 Secrets 예시: `secrets.public.example.toml`
