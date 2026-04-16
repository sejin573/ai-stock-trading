# News-Driven Korean Stock Impact Dashboard

A Streamlit portfolio project that combines Naver News Search API, KIS Open API, and pykrx to show how recent news activity may affect short-term Korean stock momentum.

## MVP scope

- Search a stock by ticker and company name
- Load Korean stock prices from KIS Open API
- Use pykrx as a backup and backtesting-friendly source
- Load recent company news from Naver News Search API
- Score each article with a lightweight rule-based sentiment model
- Detect major event tags such as earnings, product launch, lawsuit, or M&A
- Calculate an impact score with an explanation
- Visualize stock price, news volume, and article-level analysis in Streamlit

## Project structure

```text
mini_project/
|- app.py
|- app_public.py
|- requirements.txt
|- .env.example
|- README.md
|- services/
|  |- __init__.py
|  |- news_service.py
|  `- stock_service.py
|- analysis/
|  |- __init__.py
|  |- sentiment.py
|  |- event_tags.py
|  |- scoring.py
|  `- features.py
|- models/
|  |- __init__.py
|  |- train.py
|  `- predict.py
`- utils/
   |- __init__.py
   |- config.py
   `- helpers.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API keys:

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

4. Run the app:

```bash
streamlit run app.py
```

### Public portfolio app

`app_public.py` is a read-only public dashboard for Streamlit Community Cloud deployment.

- Visitors can only view the portfolio monitor.
- Buy/sell controls and strategy settings are not exposed.
- For a safe public deployment, use mock-trading secrets only.

Run it locally:

```bash
streamlit run app_public.py
```

Recommended secrets for public deployment:

```toml
NAVER_CLIENT_ID="your_naver_client_id"
NAVER_CLIENT_SECRET="your_naver_client_secret"
KIS_APP_KEY="your_kis_app_key"
KIS_APP_SECRET="your_kis_app_secret"
KIS_MOCK_APP_KEY="your_kis_mock_app_key"
KIS_MOCK_APP_SECRET="your_kis_mock_app_secret"
KIS_ACCOUNT_NO="12345678"
KIS_ACCOUNT_PRODUCT_CODE="01"
PUBLIC_APP_ENABLE_AUTO_CYCLE="true"
PUBLIC_APP_REFRESH_SECONDS="15"
PUBLIC_APP_MIN_CYCLE_INTERVAL_SECONDS="30"
PUBLIC_APP_SPOTLIGHT_SYMBOL="005930"
```

You can also start from [`secrets.public.example.toml`](./secrets.public.example.toml) and copy its values into Streamlit Community Cloud Secrets.

Notes:

- Do not upload `.env` or `.streamlit/secrets.toml` to GitHub.
- Do not use real-account trading keys in the public app.
- In Streamlit Community Cloud, set the main file to `app_public.py`.
- `PUBLIC_APP_SPOTLIGHT_SYMBOL` can be used to pin the stock chart to a specific ticker in the public app.

## Data flow

1. The user enters a ticker and company name.
2. `stock_service.py` first tries KIS Open API for Korean stock prices.
3. If KIS is unavailable or fails, `stock_service.py` falls back to pykrx.
4. `news_service.py` loads recent articles from Naver News Search API.
5. `sentiment.py` and `event_tags.py` enrich each article.
6. `features.py` aggregates news by date and merges it with stock data.
7. `scoring.py` produces an impact score, directional bias, and explanation.
8. `app.py` renders charts, metrics, and article analysis in Streamlit.

## Mock Trading Flow

The dashboard now supports mock-trading only.

1. Prepare a KIS mock trading app key/app secret.
2. Fill in `KIS_MOCK_APP_KEY`, `KIS_MOCK_APP_SECRET`, `KIS_ACCOUNT_NO`, and `KIS_ACCOUNT_PRODUCT_CODE` in `.env`.
3. Run `streamlit run app.py`.
4. Choose a stock and click `분석 실행`.
5. In the `모의투자 주문` section:
   - review the forecasted return
   - choose the quantity
   - keep `자동매도 등록` enabled if you want auto-sell tracking
   - confirm the target profit percent
6. Click `현재 종목 모의 매수`.
7. The position is stored locally in `data/mock_trade_state.json`.
8. Click `자동매도 지금 점검` to evaluate whether any saved position has reached its target.

Auto-sell rule:

- The default target is `predicted_upside * 0.5`
- Example: if the forecasted upside is `8%`, the default auto-sell target becomes `4%`

Notes:

- This is mock trading only. No real-money order should be sent from this workflow.
- Auto-sell checks are triggered from the dashboard, not by a background worker yet.

## Next steps

- Add article summarization
- Extract article body text from source URLs
- Train a next-day direction classifier
- Add multi-stock comparison
- Track model evaluation metrics
