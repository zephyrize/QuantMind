-- ============================================================
-- QuantMind OSS Database Initialization Script
-- Creates all missing tables for a fresh deployment
-- Run: docker exec -i quantmind-db psql -U quantmind -d quantmind < /tmp/quantmind_init.sql
-- ============================================================

-- ========================
-- 1. STRATEGIES (核心表 - 报错的表)
-- ========================
CREATE TABLE IF NOT EXISTS strategies (
    id                SERIAL PRIMARY KEY,
    user_id           INTEGER NOT NULL,
    name              TEXT NOT NULL,
    description       TEXT,
    strategy_type     TEXT DEFAULT 'CUSTOM',
    status            TEXT DEFAULT 'DRAFT',
    config            JSONB DEFAULT '{}',
    parameters        JSONB DEFAULT '{}',
    execution_config  JSONB DEFAULT '{}',
    code              TEXT,
    cos_url           TEXT,
    cos_key           TEXT,
    code_hash         VARCHAR(64),
    file_size         INTEGER DEFAULT 0,
    tags              TEXT[] DEFAULT '{}',
    is_public         BOOLEAN DEFAULT FALSE,
    shared_users      JSONB DEFAULT '[]',
    backtest_count    INTEGER DEFAULT 0,
    view_count        INTEGER DEFAULT 0,
    like_count        INTEGER DEFAULT 0,
    version           INTEGER DEFAULT 1,
    is_verified       BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_strategies_user_id ON strategies (user_id);
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies (status);
CREATE INDEX IF NOT EXISTS idx_strategies_is_public ON strategies (is_public) WHERE is_public = TRUE;

-- ========================
-- 2. STOCK_DAILY_LATEST (行情数据 - 分区表)
-- ========================
CREATE TABLE IF NOT EXISTS stock_daily_latest (
    trade_date        DATE NOT NULL,
    symbol            VARCHAR(20) NOT NULL,
    open              DOUBLE PRECISION,
    high              DOUBLE PRECISION,
    low               DOUBLE PRECISION,
    close             DOUBLE PRECISION,
    volume            DOUBLE PRECISION,
    amount            DOUBLE PRECISION,
    adj_factor        DOUBLE PRECISION,
    stock_name        VARCHAR(100),
    industry          VARCHAR(100),
    pe_ttm            DOUBLE PRECISION,
    pb                DOUBLE PRECISION,
    bp                DOUBLE PRECISION,
    ep_ttm            DOUBLE PRECISION,
    roe               DOUBLE PRECISION,
    ln_mv_total       DOUBLE PRECISION,
    total_mv          DOUBLE PRECISION,
    float_mv          DOUBLE PRECISION,
    turnover_rate     DOUBLE PRECISION,
    pct_change        DOUBLE PRECISION,
    is_st             INTEGER DEFAULT 0,
    idx_hs300         INTEGER DEFAULT 0,
    idx_zz1000        INTEGER DEFAULT 0,
    idx_chinext       INTEGER DEFAULT 0,
    idx_margin        INTEGER DEFAULT 0,
    idx_all           INTEGER DEFAULT 0,
    ma5               DOUBLE PRECISION,
    ma10              DOUBLE PRECISION,
    ma20              DOUBLE PRECISION,
    ma60              DOUBLE PRECISION,
    ma_gap_5          DOUBLE PRECISION,
    ma_gap_10         DOUBLE PRECISION,
    ma_gap_20         DOUBLE PRECISION,
    return_1d         DOUBLE PRECISION,
    return_3d         DOUBLE PRECISION,
    return_5d         DOUBLE PRECISION,
    return_10d        DOUBLE PRECISION,
    return_20d        DOUBLE PRECISION,
    return_60d        DOUBLE PRECISION,
    vol_std_5         DOUBLE PRECISION,
    vol_std_20        DOUBLE PRECISION,
    vol_std_60        DOUBLE PRECISION,
    vol_atr_14        DOUBLE PRECISION,
    rsi_14            DOUBLE PRECISION,
    rsi_6             DOUBLE PRECISION,
    kdj_k             DOUBLE PRECISION,
    macd_hist         DOUBLE PRECISION,
    beta_20           DOUBLE PRECISION,
    volume_ratio_5    DOUBLE PRECISION,
    volume_ratio_20   DOUBLE PRECISION,
    volume_ma_5       DOUBLE PRECISION,
    amount_ma_5       DOUBLE PRECISION,
    volume_trend_3d   BOOLEAN,
    main_flow         DOUBLE PRECISION,
    flow_net_amount   DOUBLE PRECISION,
    inst_ownership    DOUBLE PRECISION,
    profit_growth     DOUBLE PRECISION,
    listing_market    VARCHAR(20),
    listed_days       INTEGER,
    concept_ai        DOUBLE PRECISION,
    concept_chip      DOUBLE PRECISION,
    concept_new_energy DOUBLE PRECISION,
    concept_pv        DOUBLE PRECISION,
    concept_lithium   DOUBLE PRECISION,
    concept_military  DOUBLE PRECISION,
    concept_medical   DOUBLE PRECISION,
    concept_fintech   DOUBLE PRECISION,
    concept_consumption DOUBLE PRECISION,
    concept_state_owned DOUBLE PRECISION,
    consecutive_limit_up_days INTEGER DEFAULT 0,
    PRIMARY KEY (trade_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_sdl_symbol ON stock_daily_latest (symbol);
CREATE INDEX IF NOT EXISTS idx_sdl_date ON stock_daily_latest (trade_date DESC);

-- ========================
-- 3. STOCKS (股票主表)
-- NOTE: column names must match seed_a_share_stocks.py: symbol, name, exchange, industry, sector, is_active
-- ========================
CREATE TABLE IF NOT EXISTS stocks (
    symbol          VARCHAR(20) NOT NULL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    exchange        VARCHAR(20),
    industry        VARCHAR(200),
    sector          VARCHAR(200),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stocks_name ON stocks (name);
CREATE INDEX IF NOT EXISTS idx_stocks_exchange ON stocks (exchange);

-- ========================
-- 4. STOCK_INDUSTRY
-- ========================
CREATE TABLE IF NOT EXISTS stock_industry (
    id              SERIAL PRIMARY KEY,
    stock_code      VARCHAR(20) NOT NULL,
    industry_name   VARCHAR(200),
    industry_code   VARCHAR(50),
    sector_name     VARCHAR(200),
    sector_code     VARCHAR(50),
    concept_tags    TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stock_industry_code ON stock_industry (stock_code);

-- ========================
-- 5. STOCK_ALIASES (from 2026_05_25 SQL)
-- ========================
CREATE TABLE IF NOT EXISTS stock_aliases (
    id          BIGSERIAL    PRIMARY KEY,
    ticker      VARCHAR(16)  NOT NULL,
    alias       VARCHAR(64)  NOT NULL,
    alias_type  VARCHAR(16)  NOT NULL,
    priority    SMALLINT     NOT NULL DEFAULT 50,
    industry    VARCHAR(64),
    sector      VARCHAR(64),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, alias)
);

CREATE INDEX IF NOT EXISTS idx_stock_aliases_alias ON stock_aliases (alias);
CREATE INDEX IF NOT EXISTS idx_stock_aliases_ticker ON stock_aliases (ticker);

-- ========================
-- 6. NEWS_ARTICLE_ENRICHMENT (from 2026_05_25 SQL)
-- ========================
CREATE TABLE IF NOT EXISTS news_article_enrichment (
    huntly_page_id      BIGINT      PRIMARY KEY,
    tickers             TEXT[]      NOT NULL DEFAULT '{}',
    industries          TEXT[]      NOT NULL DEFAULT '{}',
    event_tags          TEXT[]      NOT NULL DEFAULT '{}',
    sentiment_score     REAL,
    sentiment_label     VARCHAR(16),
    sentiment_confidence REAL,
    enriched_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version       VARCHAR(64) NOT NULL,
    title_hash          BIGINT,
    error               TEXT,
    -- extended columns from migrations
    countries           TEXT[]      NOT NULL DEFAULT '{}',
    regions             TEXT[]      NOT NULL DEFAULT '{}',
    key_terms           TEXT[]      NOT NULL DEFAULT '{}',
    date_entities       TEXT[]      NOT NULL DEFAULT '{}',
    entity_sentiments   JSONB       NOT NULL DEFAULT '{}',
    provinces           TEXT[]      NOT NULL DEFAULT '{}',
    cities              TEXT[]      NOT NULL DEFAULT '{}',
    politicians         TEXT[]      NOT NULL DEFAULT '{}',
    visits              TEXT[]      NOT NULL DEFAULT '{}',
    departments         TEXT[]      NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_news_enrichment_tickers
    ON news_article_enrichment USING GIN (tickers);
CREATE INDEX IF NOT EXISTS idx_news_enrichment_industries
    ON news_article_enrichment USING GIN (industries);
CREATE INDEX IF NOT EXISTS idx_news_enrichment_event_tags
    ON news_article_enrichment USING GIN (event_tags);
CREATE INDEX IF NOT EXISTS idx_news_enrichment_label
    ON news_article_enrichment (sentiment_label, enriched_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_enrichment_score
    ON news_article_enrichment (sentiment_score DESC NULLS LAST);

-- ========================
-- 7. FINANCE_LEXICON (from 2026_05_25 SQL)
-- ========================
CREATE TABLE IF NOT EXISTS finance_lexicon (
    id          BIGSERIAL    PRIMARY KEY,
    term        VARCHAR(64)  NOT NULL,
    kind        VARCHAR(16)  NOT NULL,
    event_tag   VARCHAR(32),
    weight      REAL         NOT NULL DEFAULT 1.0,
    note        TEXT,
    enabled     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (term, kind)
);

CREATE INDEX IF NOT EXISTS idx_lexicon_kind_enabled ON finance_lexicon (kind, enabled);

-- ========================
-- 8. STOCK_POOL_FILES
-- ========================
CREATE TABLE IF NOT EXISTS stock_pool_files (
    id              SERIAL PRIMARY KEY,
    tenant_id       VARCHAR(50) DEFAULT 'default',
    user_id         VARCHAR(50) NOT NULL,
    pool_name       VARCHAR(200),
    session_id      VARCHAR(100),
    file_key        VARCHAR(500) NOT NULL,
    file_url        VARCHAR(1000),
    relative_path   VARCHAR(500),
    format          VARCHAR(10) DEFAULT 'csv',
    file_size       INTEGER,
    code_hash       VARCHAR(64),
    stock_count     INTEGER,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_spf_user_id ON stock_pool_files (user_id);

-- ========================
-- 9. STRATEGY_LOOP_TASKS
-- ========================
CREATE TABLE IF NOT EXISTS strategy_loop_tasks (
    task_id         TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    status          TEXT NOT NULL,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_json    JSONB,
    result_json     JSONB
);

-- ========================
-- 10. PIPELINE_RUNS
-- ========================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    status          TEXT NOT NULL,
    stage           TEXT NOT NULL,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_json    JSONB,
    result_json     JSONB
);

-- ========================
-- 11. ENGINE_FEATURE_RUNS
-- ========================
CREATE TABLE IF NOT EXISTS engine_feature_runs (
    run_id          TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    user_id         TEXT NOT NULL,
    trade_date      DATE NOT NULL,
    model_name      TEXT,
    model_version   TEXT,
    feature_version TEXT,
    feature_dim     INTEGER,
    window_start    TIMESTAMPTZ,
    window_end      TIMESTAMPTZ,
    status          TEXT NOT NULL,
    expected_symbols INTEGER,
    ready_symbols   INTEGER,
    missing_symbols INTEGER,
    source          TEXT,
    checksum        TEXT,
    quality         JSONB,
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ========================
-- 12. ENGINE_SIGNAL_SCORES
-- ========================
CREATE TABLE IF NOT EXISTS engine_signal_scores (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    trade_date      DATE NOT NULL,
    symbol          TEXT NOT NULL,
    model_version   TEXT,
    feature_version TEXT,
    light_score     DOUBLE PRECISION,
    tft_score       DOUBLE PRECISION,
    fusion_score    DOUBLE PRECISION NOT NULL,
    risk_weight     DOUBLE PRECISION DEFAULT 1.0,
    regime          TEXT DEFAULT 'normal',
    score_rank      INTEGER,
    universe_tag    TEXT,
    signal_side     TEXT,
    expected_price  DOUBLE PRECISION,
    quality         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, user_id, trade_date, symbol, model_version, feature_version, run_id)
);

CREATE INDEX IF NOT EXISTS idx_ess_run_id ON engine_signal_scores (run_id);
CREATE INDEX IF NOT EXISTS idx_ess_trade_date ON engine_signal_scores (trade_date DESC);

-- ========================
-- 13. ENGINE_DISPATCH_BATCHES
-- ========================
CREATE TABLE IF NOT EXISTS engine_dispatch_batches (
    batch_id            TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    tenant_id           TEXT NOT NULL DEFAULT 'default',
    user_id             TEXT NOT NULL,
    trade_date          DATE NOT NULL,
    strategy_id         TEXT,
    trading_mode        TEXT,
    stage               TEXT NOT NULL,
    stage_updated_at    TIMESTAMPTZ,
    total_signals       INTEGER,
    dispatched_signals  INTEGER,
    acked_signals       INTEGER,
    order_submitted_count INTEGER,
    order_filled_count  INTEGER,
    failed_count        INTEGER,
    trace_id            TEXT,
    last_error          TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ========================
-- 14. ENGINE_DISPATCH_ITEMS
-- ========================
CREATE TABLE IF NOT EXISTS engine_dispatch_items (
    id                  BIGSERIAL PRIMARY KEY,
    batch_id            TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    signal_id           TEXT,
    client_order_id     TEXT UNIQUE,
    tenant_id           TEXT NOT NULL DEFAULT 'default',
    user_id             TEXT NOT NULL,
    trade_date          DATE NOT NULL,
    symbol              TEXT NOT NULL,
    action              TEXT NOT NULL,
    quantity            DOUBLE PRECISION,
    price               DOUBLE PRECISION,
    score               DOUBLE PRECISION,
    dispatch_status     TEXT NOT NULL,
    order_id            UUID,
    exchange_order_id   TEXT,
    exchange_trade_id   TEXT,
    exec_message        TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_edi_batch_id ON engine_dispatch_items (batch_id);
CREATE INDEX IF NOT EXISTS idx_edi_symbol ON engine_dispatch_items (symbol);

-- ========================
-- 15. QM_RESEARCH_CANDIDATE_SNAPSHOT
-- ========================
CREATE TABLE IF NOT EXISTS qm_research_candidate_snapshot (
    id                      BIGSERIAL PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    user_id                 TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    model_id                TEXT,
    data_trade_date         DATE,
    prediction_trade_date   DATE,
    symbol                  TEXT NOT NULL,
    fusion_score            DOUBLE PRECISION,
    score_rank              INTEGER,
    signal_side             TEXT,
    expected_price          DOUBLE PRECISION,
    universe_tag            TEXT,
    confidence_level        TEXT,
    thesis_summary          TEXT,
    risk_flags              JSONB,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, user_id, run_id, symbol)
);

-- ========================
-- 16. QM_TRADING_AGENTS_HISTORY
-- ========================
CREATE TABLE IF NOT EXISTS qm_trading_agents_history (
    analysis_id     TEXT PRIMARY KEY,
    ticker          TEXT,
    trade_date      TEXT,
    signal          TEXT,
    llm_provider    TEXT,
    deep_think_llm  TEXT,
    quick_think_llm TEXT,
    stage_reports   JSONB,
    final_state     JSONB,
    stats           JSONB,
    elapsed_seconds DOUBLE PRECISION,
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tah_ticker ON qm_trading_agents_history (ticker);
CREATE INDEX IF NOT EXISTS idx_tah_trade_date ON qm_trading_agents_history (trade_date);

-- ========================
-- 17. QM_USER_WATCHLIST
-- ========================
CREATE TABLE IF NOT EXISTS qm_user_watchlist (
    id                  BIGSERIAL PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    stock_name          TEXT,
    added_at            TIMESTAMPTZ DEFAULT NOW(),
    source_run_id       TEXT,
    features_snapshot   JSONB,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, user_id, symbol)
);

-- ========================
-- 18. QM_USER_RESEARCH_POOL
-- ========================
CREATE TABLE IF NOT EXISTS qm_user_research_pool (
    id                  BIGSERIAL PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    stock_name          TEXT,
    added_at            TIMESTAMPTZ DEFAULT NOW(),
    source_run_id       TEXT,
    status              TEXT,
    model_id            TEXT,
    fusion_score        DOUBLE PRECISION,
    thesis_summary      TEXT,
    features_snapshot   JSONB,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, user_id, symbol)
);

-- ========================
-- 19. QM_FEATURE_CATEGORY
-- ========================
CREATE TABLE IF NOT EXISTS qm_feature_category (
    category_id     VARCHAR PRIMARY KEY,
    category_name   VARCHAR,
    sort_order      INTEGER,
    description     TEXT
);

-- ========================
-- 20. QM_FEATURE_DEFINITION
-- ========================
CREATE TABLE IF NOT EXISTS qm_feature_definition (
    feature_id          UUID,
    feature_key         VARCHAR PRIMARY KEY,
    feature_name        VARCHAR,
    formula             TEXT,
    category_id         VARCHAR REFERENCES qm_feature_category (category_id),
    source_table_fields TEXT,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ========================
-- 21. QM_FEATURE_SET_VERSION
-- ========================
CREATE TABLE IF NOT EXISTS qm_feature_set_version (
    version_id      VARCHAR PRIMARY KEY,
    version_name    VARCHAR,
    status          VARCHAR,
    feature_count   INTEGER,
    effective_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ========================
-- 22. QM_FEATURE_SET_ITEM
-- ========================
CREATE TABLE IF NOT EXISTS qm_feature_set_item (
    id              SERIAL PRIMARY KEY,
    version_id      VARCHAR REFERENCES qm_feature_set_version (version_id),
    category_id     VARCHAR REFERENCES qm_feature_category (category_id),
    feature_key     VARCHAR REFERENCES qm_feature_definition (feature_key),
    order_no        INTEGER,
    enabled         BOOLEAN DEFAULT TRUE,
    UNIQUE (version_id, feature_key)
);

-- ========================
-- 23. QM_MARKET_CALENDAR_DAY
-- ========================
CREATE TABLE IF NOT EXISTS qm_market_calendar_day (
    market          VARCHAR(32) NOT NULL,
    trade_date      DATE NOT NULL,
    is_trading_day  BOOLEAN NOT NULL,
    timezone        VARCHAR(64) NOT NULL DEFAULT 'Asia/Shanghai',
    source          VARCHAR(64) NOT NULL DEFAULT 'manual',
    version         VARCHAR(64),
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL DEFAULT '*',
    metadata_json   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (market, trade_date, tenant_id, user_id)
);

-- ========================
-- 24. QM_MARKET_TRADING_SESSION
-- ========================
CREATE TABLE IF NOT EXISTS qm_market_trading_session (
    market          VARCHAR(32) NOT NULL,
    session_name    VARCHAR(64) NOT NULL,
    start_time      TIME NOT NULL,
    end_time        TIME NOT NULL,
    cross_day       BOOLEAN NOT NULL DEFAULT FALSE,
    trade_date_rule VARCHAR(64) NOT NULL DEFAULT 'TRADE_DATE',
    timezone        VARCHAR(64) NOT NULL DEFAULT 'Asia/Shanghai',
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL DEFAULT '*',
    metadata_json   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (market, session_name, tenant_id, user_id)
);

-- ========================
-- 25. QM_MARKET_CALENDAR_EXCEPTION
-- ========================
CREATE TABLE IF NOT EXISTS qm_market_calendar_exception (
    id              BIGSERIAL PRIMARY KEY,
    market          VARCHAR(32) NOT NULL,
    trade_date      DATE NOT NULL,
    action          VARCHAR(16) NOT NULL,
    reason          TEXT,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL DEFAULT '*',
    approved_by     VARCHAR(128),
    metadata_json   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ========================
-- 26. QM_MARKET_CALENDAR_VERSION
-- ========================
CREATE TABLE IF NOT EXISTS qm_market_calendar_version (
    market          VARCHAR(32) NOT NULL,
    year            INTEGER NOT NULL,
    checksum        VARCHAR(128) NOT NULL,
    status          VARCHAR(32) NOT NULL DEFAULT 'draft',
    source          VARCHAR(64),
    published_at    TIMESTAMPTZ,
    metadata_json   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (market, year)
);

-- ========================
-- 27. DATA_QUALITY_ALERTS (from 2026_05_24 SQL)
-- ========================
CREATE TABLE IF NOT EXISTS data_quality_alerts (
    id              BIGSERIAL PRIMARY KEY,
    alert_type      VARCHAR(32) NOT NULL,
    severity        VARCHAR(16) NOT NULL,
    market          VARCHAR(8),
    field           VARCHAR(48),
    source          VARCHAR(32),
    symbol          VARCHAR(32),
    trade_date      DATE,
    message         TEXT NOT NULL,
    details         JSONB,
    acknowledged    BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_by VARCHAR(64),
    acknowledged_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dqa_created_at ON data_quality_alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dqa_unack_severity ON data_quality_alerts (acknowledged, severity, created_at DESC) WHERE acknowledged = FALSE;
CREATE INDEX IF NOT EXISTS idx_dqa_market_field ON data_quality_alerts (market, field, created_at DESC);

-- ========================
-- 28. REAL_ACCOUNT_BASELINES
-- ========================
CREATE TABLE IF NOT EXISTS real_account_baselines (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    account_id      TEXT NOT NULL,
    initial_equity  DOUBLE PRECISION,
    first_snapshot_at TIMESTAMPTZ,
    source          TEXT DEFAULT 'qmt_bridge_first_report',
    UNIQUE (tenant_id, user_id, account_id)
);

-- ========================
-- 29. TRADING_CALENDAR (simple version from SQL)
-- ========================
CREATE TABLE IF NOT EXISTS trading_calendar (
    market         VARCHAR(8)  NOT NULL,
    trade_date     DATE        NOT NULL,
    is_trading     BOOLEAN     NOT NULL,
    is_half_day    BOOLEAN     NOT NULL DEFAULT FALSE,
    note           TEXT,
    source         VARCHAR(32),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (market, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_trading_calendar_market_trading ON trading_calendar (market, is_trading, trade_date);

-- ========================
-- 30. AI_STRATEGIES (legacy table)
-- ========================
CREATE TABLE IF NOT EXISTS ai_strategies (
    id              SERIAL PRIMARY KEY,
    strategy_id     VARCHAR(64) UNIQUE,
    user_id         VARCHAR(64),
    name            VARCHAR(255),
    description     TEXT,
    market          VARCHAR(32),
    risk_level      VARCHAR(16),
    provider        VARCHAR(32),
    code            TEXT,
    cos_file_key    VARCHAR(500),
    cos_file_url    VARCHAR(1000),
    factors         TEXT,
    risk_controls   TEXT,
    assumptions     TEXT,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_strategies_user_id ON ai_strategies (user_id);

-- ========================
-- 31. USER_STRATEGIES (legacy table for migration)
-- ========================
CREATE TABLE IF NOT EXISTS user_strategies (
    id              SERIAL PRIMARY KEY,
    user_id         VARCHAR(64) NOT NULL,
    strategy_name   VARCHAR(255),
    description     TEXT,
    conditions      JSONB,
    stock_pool      JSONB,
    position_config JSONB,
    style           VARCHAR(32),
    risk_config     JSONB,
    cos_url         TEXT,
    file_size       INTEGER,
    code_hash       VARCHAR(64),
    qlib_validated  BOOLEAN DEFAULT FALSE,
    validation_result JSONB,
    tags            TEXT[],
    is_public       BOOLEAN DEFAULT FALSE,
    downloads       INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_us_user_id ON user_strategies (user_id);

-- ========================
-- TRADE ENUMS (needed for orders/trades tables)
-- ========================
DO $$ BEGIN
    -- Create enums if they don't exist
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'orderside') THEN
        CREATE TYPE orderside AS ENUM ('buy', 'sell');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tradeaction') THEN
        CREATE TYPE tradeaction AS ENUM ('buy_to_open', 'sell_to_close', 'sell_to_open', 'buy_to_close');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'positionside') THEN
        CREATE TYPE positionside AS ENUM ('long', 'short');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ordertype') THEN
        CREATE TYPE ordertype AS ENUM ('market', 'limit', 'stop', 'stop_limit');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tradingmode') THEN
        CREATE TYPE tradingmode AS ENUM ('SIMULATION', 'SHADOW', 'REAL');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'orderstatus') THEN
        CREATE TYPE orderstatus AS ENUM ('pending', 'submitted', 'partially_filled', 'filled', 'cancelled', 'rejected', 'expired');
    END IF;
END $$;

-- ========================
-- 32. ORDERS
-- ========================
CREATE TABLE IF NOT EXISTS orders (
    id              SERIAL PRIMARY KEY,
    order_id        UUID NOT NULL UNIQUE,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(32) NOT NULL,
    portfolio_id    INTEGER NOT NULL,
    strategy_id     INTEGER,
    symbol          VARCHAR(20) NOT NULL,
    symbol_name     VARCHAR(50),
    side            orderside NOT NULL,
    trade_action    tradeaction,
    position_side   positionside NOT NULL,
    is_margin_trade BOOLEAN NOT NULL DEFAULT FALSE,
    order_type      ordertype NOT NULL,
    trading_mode    tradingmode NOT NULL,
    status          orderstatus NOT NULL,
    quantity        FLOAT NOT NULL,
    filled_quantity FLOAT NOT NULL DEFAULT 0,
    price           FLOAT,
    stop_price      FLOAT,
    average_price   FLOAT,
    order_value     FLOAT NOT NULL DEFAULT 0,
    filled_value    FLOAT NOT NULL DEFAULT 0,
    commission      FLOAT NOT NULL DEFAULT 0,
    submitted_at    TIMESTAMP,
    filled_at       TIMESTAMP,
    cancelled_at    TIMESTAMP,
    expired_at      TIMESTAMP,
    client_order_id VARCHAR(100) UNIQUE,
    exchange_order_id VARCHAR(100),
    remarks         VARCHAR(500),
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ========================
-- 33. TRADES
-- ========================
CREATE TABLE IF NOT EXISTS trades (
    id              SERIAL PRIMARY KEY,
    trade_id        UUID NOT NULL UNIQUE,
    order_id        UUID NOT NULL REFERENCES orders (order_id),
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(32) NOT NULL,
    portfolio_id    INTEGER NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    symbol_name     VARCHAR(50),
    side            orderside NOT NULL,
    trade_action    tradeaction,
    position_side   positionside NOT NULL,
    is_margin_trade BOOLEAN NOT NULL DEFAULT FALSE,
    trading_mode    tradingmode NOT NULL,
    quantity        FLOAT NOT NULL,
    price           FLOAT NOT NULL,
    trade_value     FLOAT NOT NULL,
    commission      FLOAT NOT NULL DEFAULT 0,
    stamp_duty      FLOAT NOT NULL DEFAULT 0,
    transfer_fee    FLOAT NOT NULL DEFAULT 0,
    total_fee       FLOAT NOT NULL DEFAULT 0,
    executed_at     TIMESTAMP NOT NULL,
    exchange_trade_id VARCHAR(100),
    exchange_name   VARCHAR(50),
    remarks         VARCHAR(500),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ========================
-- 34. PORTFOLIOS
-- ========================
CREATE TABLE IF NOT EXISTS portfolios (
    id                  SERIAL PRIMARY KEY,
    tenant_id           VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id             VARCHAR(32) NOT NULL,
    name                VARCHAR(100) NOT NULL,
    description         TEXT,
    initial_capital     NUMERIC(20, 2) NOT NULL DEFAULT 0,
    current_capital     NUMERIC(20, 2) NOT NULL DEFAULT 0,
    available_cash      NUMERIC(20, 2) NOT NULL DEFAULT 0,
    frozen_cash         NUMERIC(20, 2) NOT NULL DEFAULT 0,
    total_value         NUMERIC(20, 2) NOT NULL DEFAULT 0,
    total_pnl           NUMERIC(20, 2) NOT NULL DEFAULT 0,
    total_return        NUMERIC(10, 4) NOT NULL DEFAULT 0,
    daily_pnl           NUMERIC(20, 2) NOT NULL DEFAULT 0,
    daily_return        NUMERIC(10, 4) NOT NULL DEFAULT 0,
    yesterday_total_value NUMERIC(20, 2) NOT NULL DEFAULT 0,
    max_drawdown        NUMERIC(10, 4) NOT NULL DEFAULT 0,
    sharpe_ratio        NUMERIC(10, 4),
    volatility          NUMERIC(10, 4),
    status              VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    trading_mode        tradingmode NOT NULL DEFAULT 'SIMULATION',
    broker_type         VARCHAR(32),
    broker_account_id   VARCHAR(64),
    broker_params       JSONB,
    strategy_id         INTEGER,
    real_trading_id     VARCHAR(50),
    run_status          VARCHAR(20) NOT NULL DEFAULT 'STOPPED',
    is_deleted          BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT check_initial_capital_positive CHECK (initial_capital >= 0),
    CONSTRAINT check_available_cash_positive CHECK (available_cash >= 0)
);

-- ========================
-- 35. POSITIONS
-- ========================
CREATE TABLE IF NOT EXISTS positions (
    id                  SERIAL PRIMARY KEY,
    portfolio_id        INTEGER NOT NULL REFERENCES portfolios (id),
    symbol              VARCHAR(20) NOT NULL,
    symbol_name         VARCHAR(100),
    exchange            VARCHAR(20),
    side                VARCHAR(20) NOT NULL DEFAULT 'LONG',
    quantity            INTEGER NOT NULL DEFAULT 0,
    available_quantity  INTEGER NOT NULL DEFAULT 0,
    frozen_quantity     INTEGER NOT NULL DEFAULT 0,
    avg_cost            NUMERIC(20, 4) NOT NULL DEFAULT 0,
    total_cost          NUMERIC(20, 2) NOT NULL DEFAULT 0,
    current_price       NUMERIC(20, 4) NOT NULL DEFAULT 0,
    market_value        NUMERIC(20, 2) NOT NULL DEFAULT 0,
    unrealized_pnl      NUMERIC(20, 2) NOT NULL DEFAULT 0,
    unrealized_pnl_rate NUMERIC(10, 4) NOT NULL DEFAULT 0,
    realized_pnl        NUMERIC(20, 2) NOT NULL DEFAULT 0,
    weight              NUMERIC(10, 4) NOT NULL DEFAULT 0,
    status              VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    opened_at           TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    closed_at           TIMESTAMP,
    CONSTRAINT check_quantity_positive CHECK (quantity >= 0),
    CONSTRAINT check_available_quantity_positive CHECK (available_quantity >= 0)
);

-- ========================
-- 36. POSITION_HISTORY
-- ========================
CREATE TABLE IF NOT EXISTS position_history (
    id              SERIAL PRIMARY KEY,
    position_id     INTEGER NOT NULL REFERENCES positions (id),
    action          VARCHAR(20) NOT NULL,
    quantity_change INTEGER NOT NULL,
    price           NUMERIC(20, 4) NOT NULL,
    amount          NUMERIC(20, 2) NOT NULL,
    quantity_after  INTEGER NOT NULL,
    avg_cost_after  NUMERIC(20, 4) NOT NULL,
    note            TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ========================
-- 37. PORTFOLIO_SNAPSHOTS
-- ========================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              SERIAL PRIMARY KEY,
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios (id),
    snapshot_date   TIMESTAMP NOT NULL,
    total_value     NUMERIC(20, 2) NOT NULL DEFAULT 0,
    available_cash  NUMERIC(20, 2) NOT NULL DEFAULT 0,
    market_value    NUMERIC(20, 2) NOT NULL DEFAULT 0,
    total_pnl       NUMERIC(20, 2) NOT NULL DEFAULT 0,
    total_return    NUMERIC(10, 4) NOT NULL DEFAULT 0,
    daily_pnl       NUMERIC(20, 2) NOT NULL DEFAULT 0,
    daily_return    NUMERIC(10, 4) NOT NULL DEFAULT 0,
    max_drawdown    NUMERIC(10, 4) NOT NULL DEFAULT 0,
    sharpe_ratio    NUMERIC(10, 4),
    volatility      NUMERIC(10, 4),
    position_count  INTEGER NOT NULL DEFAULT 0,
    is_settlement   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ========================
-- 38. RISK_RULES
-- ========================
CREATE TABLE IF NOT EXISTS risk_rules (
    id              SERIAL PRIMARY KEY,
    rule_name       VARCHAR(100) NOT NULL,
    rule_type       VARCHAR(50) NOT NULL,
    description     VARCHAR(500),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    parameters      JSONB NOT NULL DEFAULT '{}',
    applies_to_all  BOOLEAN NOT NULL DEFAULT TRUE,
    user_ids        JSONB,
    priority        INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ========================
-- 39. REAL_ACCOUNT_SNAPSHOTS
-- ========================
CREATE TABLE IF NOT EXISTS real_account_snapshots (
    id              SERIAL PRIMARY KEY,
    tenant_id       VARCHAR(50) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(50) NOT NULL,
    account_id      VARCHAR(64) NOT NULL,
    snapshot_at     TIMESTAMP NOT NULL,
    snapshot_date   DATE NOT NULL,
    snapshot_month  VARCHAR(7) NOT NULL,
    total_asset     FLOAT NOT NULL DEFAULT 0,
    cash            FLOAT NOT NULL DEFAULT 0,
    market_value    FLOAT NOT NULL DEFAULT 0,
    today_pnl_raw   FLOAT NOT NULL DEFAULT 0,
    total_pnl_raw   FLOAT NOT NULL DEFAULT 0,
    floating_pnl_raw FLOAT NOT NULL DEFAULT 0,
    source          VARCHAR(32) NOT NULL DEFAULT 'qmt',
    payload_json    JSONB NOT NULL DEFAULT '{}'
);

-- ========================
-- 40. REAL_TRADING_PREFLIGHT_SNAPSHOTS
-- ========================
CREATE TABLE IF NOT EXISTS real_trading_preflight_snapshots (
    id                  SERIAL PRIMARY KEY,
    tenant_id           VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id             VARCHAR(64) NOT NULL,
    trading_mode        VARCHAR(16) NOT NULL,
    snapshot_date       DATE NOT NULL,
    ready               BOOLEAN NOT NULL DEFAULT FALSE,
    total_checks        INTEGER NOT NULL DEFAULT 0,
    passed_checks       INTEGER NOT NULL DEFAULT 0,
    required_failed_count INTEGER NOT NULL DEFAULT 0,
    run_count           INTEGER NOT NULL DEFAULT 0,
    failed_required_keys JSONB NOT NULL DEFAULT '[]',
    checks              JSONB NOT NULL DEFAULT '[]',
    source              VARCHAR(32) NOT NULL DEFAULT 'preflight',
    last_checked_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, user_id, trading_mode, snapshot_date)
);

-- ========================
-- 41. REAL_ACCOUNT_LEDGER_DAILY_SNAPSHOTS
-- ========================
CREATE TABLE IF NOT EXISTS real_account_ledger_daily_snapshots (
    id              SERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL,
    account_id      VARCHAR(64) NOT NULL,
    snapshot_date   DATE NOT NULL,
    last_snapshot_at TIMESTAMP NOT NULL DEFAULT NOW(),
    initial_equity  FLOAT NOT NULL DEFAULT 0,
    day_open_equity FLOAT NOT NULL DEFAULT 0,
    month_open_equity FLOAT NOT NULL DEFAULT 0,
    total_asset     FLOAT NOT NULL DEFAULT 0,
    cash            FLOAT NOT NULL DEFAULT 0,
    market_value    FLOAT NOT NULL DEFAULT 0,
    today_pnl_raw   FLOAT NOT NULL DEFAULT 0,
    monthly_pnl_raw FLOAT NOT NULL DEFAULT 0,
    total_pnl_raw   FLOAT NOT NULL DEFAULT 0,
    floating_pnl_raw FLOAT NOT NULL DEFAULT 0,
    daily_return_pct FLOAT NOT NULL DEFAULT 0,
    total_return_pct FLOAT NOT NULL DEFAULT 0,
    position_count  INTEGER NOT NULL DEFAULT 0,
    source          VARCHAR(32) NOT NULL DEFAULT 'qmt',
    payload_json    JSONB NOT NULL DEFAULT '{}',
    UNIQUE (tenant_id, user_id, account_id, snapshot_date)
);

-- ========================
-- 42. SIM_ORDERS
-- ========================
CREATE TABLE IF NOT EXISTS sim_orders (
    id              SERIAL PRIMARY KEY,
    order_id        UUID NOT NULL UNIQUE,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         INTEGER NOT NULL,
    portfolio_id    INTEGER NOT NULL,
    strategy_id     INTEGER,
    symbol          VARCHAR(20) NOT NULL,
    side            orderside NOT NULL,
    order_type      ordertype NOT NULL,
    trading_mode    tradingmode NOT NULL DEFAULT 'SIMULATION',
    status          orderstatus NOT NULL,
    quantity        FLOAT NOT NULL,
    filled_quantity FLOAT NOT NULL DEFAULT 0,
    price           FLOAT,
    average_price   FLOAT,
    order_value     FLOAT NOT NULL DEFAULT 0,
    filled_value    FLOAT NOT NULL DEFAULT 0,
    commission      FLOAT NOT NULL DEFAULT 0,
    submitted_at    TIMESTAMPTZ,
    filled_at       TIMESTAMPTZ,
    cancelled_at    TIMESTAMPTZ,
    execution_model VARCHAR(32) NOT NULL DEFAULT 'next_bar_open',
    price_source    VARCHAR(64),
    remarks         VARCHAR(500),
    version         INTEGER NOT NULL DEFAULT 1,
    total_fee       FLOAT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ========================
-- 43. SIM_TRADES
-- ========================
CREATE TABLE IF NOT EXISTS sim_trades (
    id              SERIAL PRIMARY KEY,
    trade_id        UUID NOT NULL UNIQUE,
    order_id        UUID NOT NULL REFERENCES sim_orders (order_id),
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         INTEGER NOT NULL,
    portfolio_id    INTEGER NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    side            orderside NOT NULL,
    trading_mode    tradingmode NOT NULL DEFAULT 'SIMULATION',
    quantity        FLOAT NOT NULL,
    price           FLOAT NOT NULL,
    trade_value     FLOAT NOT NULL DEFAULT 0,
    commission      FLOAT NOT NULL DEFAULT 0,
    stamp_duty      FLOAT NOT NULL DEFAULT 0,
    transfer_fee    FLOAT NOT NULL DEFAULT 0,
    total_fee       FLOAT NOT NULL DEFAULT 0,
    executed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    price_source    VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ========================
-- 44. SIMULATION_FUND_SNAPSHOTS
-- ========================
CREATE TABLE IF NOT EXISTS simulation_fund_snapshots (
    id              SERIAL PRIMARY KEY,
    tenant_id       VARCHAR(50) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(50) NOT NULL,
    snapshot_date   DATE NOT NULL,
    total_asset     FLOAT NOT NULL DEFAULT 0,
    available_balance FLOAT NOT NULL DEFAULT 0,
    frozen_balance  FLOAT NOT NULL DEFAULT 0,
    market_value    FLOAT NOT NULL DEFAULT 0,
    initial_capital FLOAT NOT NULL DEFAULT 0,
    total_pnl       FLOAT NOT NULL DEFAULT 0,
    today_pnl       FLOAT NOT NULL DEFAULT 0,
    source          VARCHAR(64) NOT NULL DEFAULT 'sim',
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, user_id, snapshot_date)
);

-- ========================
-- 45. KLINES
-- ========================
CREATE TABLE IF NOT EXISTS klines (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    interval        VARCHAR(10) NOT NULL,
    timestamp       TIMESTAMP NOT NULL,
    open_price      FLOAT NOT NULL,
    high_price      FLOAT NOT NULL,
    low_price       FLOAT NOT NULL,
    close_price     FLOAT NOT NULL,
    volume          INTEGER NOT NULL,
    amount          FLOAT,
    change          FLOAT,
    change_percent  FLOAT,
    turnover_rate   FLOAT,
    data_source     VARCHAR(20),
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (symbol, interval, timestamp)
);

-- ========================
-- 46. QUOTES
-- ========================
CREATE TABLE IF NOT EXISTS quotes (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    open_price      FLOAT,
    high_price      FLOAT,
    low_price       FLOAT,
    close_price     FLOAT,
    current_price   FLOAT NOT NULL,
    volume          INTEGER,
    amount          FLOAT,
    pre_close       FLOAT,
    change          FLOAT,
    change_percent  FLOAT,
    bid1_price      FLOAT,
    bid1_volume     INTEGER,
    bid2_price      FLOAT,
    bid2_volume     INTEGER,
    bid3_price      FLOAT,
    bid3_volume     INTEGER,
    bid4_price      FLOAT,
    bid4_volume     INTEGER,
    bid5_price      FLOAT,
    bid5_volume     INTEGER,
    ask1_price      FLOAT,
    ask1_volume     INTEGER,
    ask2_price      FLOAT,
    ask2_volume     INTEGER,
    ask3_price      FLOAT,
    ask3_volume     INTEGER,
    ask4_price      FLOAT,
    ask4_volume     INTEGER,
    ask5_price      FLOAT,
    ask5_volume     INTEGER,
    data_source     VARCHAR(20),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ========================
-- 47. QUOTE_DAILY_SUMMARIES
-- ========================
CREATE TABLE IF NOT EXISTS quote_daily_summaries (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    trade_date      DATE NOT NULL,
    open_price      FLOAT,
    high_price      FLOAT,
    low_price       FLOAT,
    close_price     FLOAT,
    avg_price       FLOAT,
    volume          BIGINT,
    volume_sum      BIGINT,
    amount          FLOAT,
    amount_sum      FLOAT,
    quote_count     INTEGER,
    pre_close       FLOAT,
    change_pct      FLOAT,
    turnover_rate   FLOAT,
    data_source     VARCHAR(32) NOT NULL DEFAULT 'remote_redis',
    first_quote_at  TIMESTAMPTZ,
    last_quote_at   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (trade_date, symbol, data_source)
);

-- ========================
-- 48. COMMUNITY_POSTS
-- ========================
CREATE TABLE IF NOT EXISTS community_posts (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    author_id       VARCHAR(64) NOT NULL,
    title           VARCHAR(256) NOT NULL,
    content         TEXT NOT NULL,
    category        VARCHAR(64),
    tags            JSONB DEFAULT '[]',
    media           JSONB DEFAULT '[]',
    excerpt         TEXT,
    views           INTEGER DEFAULT 0,
    likes           INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    collections     INTEGER DEFAULT 0,
    pinned          BOOLEAN DEFAULT FALSE,
    featured        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    last_comment_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cp_tenant_id ON community_posts (tenant_id);
CREATE INDEX IF NOT EXISTS idx_cp_author_id ON community_posts (author_id);
CREATE INDEX IF NOT EXISTS idx_cp_category ON community_posts (tenant_id, category);

-- ========================
-- 49. COMMUNITY_COMMENTS
-- ========================
CREATE TABLE IF NOT EXISTS community_comments (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    post_id         BIGINT NOT NULL REFERENCES community_posts (id),
    author_id       VARCHAR(64) NOT NULL,
    content         TEXT NOT NULL,
    parent_id       BIGINT,
    reply_to_id     BIGINT,
    likes           INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cc_post_id ON community_comments (post_id);
CREATE INDEX IF NOT EXISTS idx_cc_author_id ON community_comments (author_id);

-- ========================
-- 50. COMMUNITY_INTERACTIONS
-- ========================
CREATE TABLE IF NOT EXISTS community_interactions (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL,
    post_id         BIGINT,
    comment_id      BIGINT,
    type            VARCHAR(32) NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (tenant_id, user_id, post_id, comment_id, type)
);

-- ========================
-- 51. COMMUNITY_AUTHOR_FOLLOWS
-- ========================
CREATE TABLE IF NOT EXISTS community_author_follows (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    follower_user_id VARCHAR(64) NOT NULL,
    author_user_id  VARCHAR(64) NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (tenant_id, follower_user_id, author_user_id)
);

-- ========================
-- 52. COMMUNITY_AUDIT_LOGS
-- ========================
CREATE TABLE IF NOT EXISTS community_audit_logs (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL,
    action          VARCHAR(64) NOT NULL,
    entity_type     VARCHAR(64) NOT NULL,
    entity_id       VARCHAR(64),
    ip              VARCHAR(64),
    user_agent      VARCHAR(256),
    meta            JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ========================
-- 53. ADMIN_MODELS
-- ========================
CREATE TABLE IF NOT EXISTS admin_models (
    id              SERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL,
    name            VARCHAR(128) NOT NULL,
    description     TEXT,
    source_type     VARCHAR(32) NOT NULL,
    start_date      TIMESTAMP,
    end_date        TIMESTAMP,
    config          JSONB,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- ========================
-- 54. ADMIN_DATA_FILES
-- ========================
CREATE TABLE IF NOT EXISTS admin_data_files (
    id              SERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    data_source_id  INTEGER REFERENCES admin_models (id) ON DELETE CASCADE,
    filename        VARCHAR(255) NOT NULL,
    file_size       INTEGER,
    status          VARCHAR(32),
    meta            JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ========================
-- 55. ADMIN_TRAINING_JOBS
-- ========================
CREATE TABLE IF NOT EXISTS admin_training_jobs (
    id              VARCHAR(64) PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         VARCHAR(64) NOT NULL,
    status          VARCHAR(32),
    instance_id     VARCHAR(64),
    request_payload JSONB,
    logs            TEXT,
    result          JSONB,
    progress        INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- ========================
-- 56. LOGIN_DEVICES
-- ========================
CREATE TABLE IF NOT EXISTS login_devices (
    id              SERIAL PRIMARY KEY,
    user_id         VARCHAR(64) NOT NULL,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    device_id       VARCHAR(128) NOT NULL,
    device_name     VARCHAR(128),
    device_type     VARCHAR(32),
    os              VARCHAR(64),
    browser         VARCHAR(64),
    ip_address      VARCHAR(64),
    location        VARCHAR(128),
    is_trusted      BOOLEAN DEFAULT FALSE,
    is_active       BOOLEAN DEFAULT TRUE,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ,
    last_location_change TIMESTAMPTZ
);

-- ========================
-- REPLAY (时光回放：模拟盘历史单步推演)
-- 与 sim_orders/sim_trades 刻意分表：会话生命周期独立，
-- 且 trade_date 记录的是「模拟交易日」而非墙钟时间。
-- ========================
CREATE TABLE IF NOT EXISTS replay_sessions (
    session_id      UUID PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'default',
    user_id         INTEGER NOT NULL,
    name            VARCHAR(128) NOT NULL DEFAULT '',
    model_id        VARCHAR(128),
    strategy_params JSONB NOT NULL DEFAULT '{}'::jsonb,
    initial_cash    FLOAT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    cursor_date     DATE,
    next_date       DATE,
    sessions_total  INTEGER NOT NULL DEFAULT 0,
    sessions_done   INTEGER NOT NULL DEFAULT 0,
    status          VARCHAR(20) NOT NULL DEFAULT 'creating',
    signal_progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    auto_trade      BOOLEAN NOT NULL DEFAULT TRUE,
    stop_loss_pct   FLOAT,
    pending_orders  JSONB,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_replay_session_scope_status
    ON replay_sessions (tenant_id, user_id, status);

CREATE TABLE IF NOT EXISTS replay_orders (
    id              SERIAL PRIMARY KEY,
    order_id        UUID NOT NULL UNIQUE,
    session_id      UUID NOT NULL REFERENCES replay_sessions(session_id) ON DELETE CASCADE,
    trade_date      DATE NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    order_type      VARCHAR(10) NOT NULL DEFAULT 'market',
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    origin          VARCHAR(20) NOT NULL DEFAULT 'signal',
    quantity        FLOAT NOT NULL,
    filled_quantity FLOAT NOT NULL DEFAULT 0,
    price           FLOAT,
    average_price   FLOAT,
    filled_value    FLOAT NOT NULL DEFAULT 0,
    total_fee       FLOAT NOT NULL DEFAULT 0,
    reject_reason   VARCHAR(200),
    price_source    VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_replay_order_session_date
    ON replay_orders (session_id, trade_date);

CREATE TABLE IF NOT EXISTS replay_trades (
    id              SERIAL PRIMARY KEY,
    trade_id        UUID NOT NULL UNIQUE,
    session_id      UUID NOT NULL REFERENCES replay_sessions(session_id) ON DELETE CASCADE,
    order_id        UUID NOT NULL REFERENCES replay_orders(order_id) ON DELETE CASCADE,
    trade_date      DATE NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    origin          VARCHAR(20) NOT NULL DEFAULT 'signal',
    quantity        FLOAT NOT NULL,
    price           FLOAT NOT NULL,
    trade_value     FLOAT NOT NULL,
    commission      FLOAT NOT NULL DEFAULT 0,
    stamp_duty      FLOAT NOT NULL DEFAULT 0,
    transfer_fee    FLOAT NOT NULL DEFAULT 0,
    total_fee       FLOAT NOT NULL DEFAULT 0,
    price_source    VARCHAR(64),
    avg_cost_before FLOAT,
    realized_pnl    FLOAT,
    holding_days    INTEGER,
    executed_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_replay_trade_session_date
    ON replay_trades (session_id, trade_date);

CREATE TABLE IF NOT EXISTS replay_equity_snapshots (
    id              SERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES replay_sessions(session_id) ON DELETE CASCADE,
    trade_date      DATE NOT NULL,
    cash            FLOAT NOT NULL DEFAULT 0,
    market_value    FLOAT NOT NULL DEFAULT 0,
    total_asset     FLOAT NOT NULL DEFAULT 0,
    day_pnl         FLOAT NOT NULL DEFAULT 0,
    cum_pnl         FLOAT NOT NULL DEFAULT 0,
    realized_pnl_cum FLOAT NOT NULL DEFAULT 0,
    unrealized_pnl  FLOAT NOT NULL DEFAULT 0,
    position_count  INTEGER NOT NULL DEFAULT 0,
    positions       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_replay_equity_session_date UNIQUE (session_id, trade_date)
);

CREATE TABLE IF NOT EXISTS replay_signals (
    id              SERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES replay_sessions(session_id) ON DELETE CASCADE,
    trade_date      DATE NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    score           FLOAT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_replay_signal_session_date_symbol UNIQUE (session_id, trade_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_replay_signal_session_date
    ON replay_signals (session_id, trade_date);

-- ========================
-- DONE - 所有缺失表已创建
-- ========================
