-- 资讯文章 enrichment 表
-- 文章原始数据由 Huntly 持有(SQLite)；这里只存 QuantMind 对每篇文章的金融加注。
-- huntly_page_id = Huntly page.id (BIGINT)
-- 每次 enrich 完成后 upsert 一行；model_version 用于强制重跑老数据。

CREATE TABLE IF NOT EXISTS news_article_enrichment (
    huntly_page_id      BIGINT      PRIMARY KEY,
    tickers             TEXT[]      NOT NULL DEFAULT '{}',     -- ["600519.SH", "000858.SZ"]
    industries          TEXT[]      NOT NULL DEFAULT '{}',     -- ["白酒", "食品饮料"]
    event_tags          TEXT[]      NOT NULL DEFAULT '{}',     -- ["回购", "业绩超预期", "重组失败"]
    sentiment_score     REAL,                                  -- [-1.000, 1.000]; null = 模型未跑
    sentiment_label     VARCHAR(16),                           -- bullish / bearish / neutral
    sentiment_confidence REAL,                                 -- [0,1]
    enriched_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version       VARCHAR(64) NOT NULL,                  -- 形如 "ac-v1+finbert-zh-v1"
    title_hash          BIGINT,                                -- 用于检测文章被改写
    error               TEXT                                   -- enrich 失败时记录
);

-- GIN 索引：按 ticker 或行业过滤文章列表
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


-- 股票别名表（每只股票多个匹配字符串 → ticker）
-- Aho-Corasick 加载时一次性 scan 全表。
-- priority 用于歧义消解：完整公司名 > 缩写 > 拼音。
CREATE TABLE IF NOT EXISTS stock_aliases (
    id          BIGSERIAL    PRIMARY KEY,
    ticker      VARCHAR(16)  NOT NULL,             -- "600519.SH"
    alias       VARCHAR(64)  NOT NULL,             -- "贵州茅台" / "茅台" / "Maotai"
    alias_type  VARCHAR(16)  NOT NULL,             -- name/short/pinyin/english/code
    priority    SMALLINT     NOT NULL DEFAULT 50,  -- 高优先匹配先
    industry    VARCHAR(64),                       -- 申万一级
    sector      VARCHAR(64),                       -- 申万二级
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, alias)
);

CREATE INDEX IF NOT EXISTS idx_stock_aliases_alias ON stock_aliases (alias);
CREATE INDEX IF NOT EXISTS idx_stock_aliases_ticker ON stock_aliases (ticker);


-- 金融情感 / 事件 词典表（运行时缓存到内存，可用 SQL 在线增删）
CREATE TABLE IF NOT EXISTS finance_lexicon (
    id          BIGSERIAL    PRIMARY KEY,
    term        VARCHAR(64)  NOT NULL,
    kind        VARCHAR(16)  NOT NULL,             -- sentiment_pos / sentiment_neg / event
    event_tag   VARCHAR(32),                       -- kind=event 时填，比如 "回购"
    weight      REAL         NOT NULL DEFAULT 1.0, -- 情感词的强度系数
    note        TEXT,
    enabled     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (term, kind)
);

CREATE INDEX IF NOT EXISTS idx_lexicon_kind_enabled ON finance_lexicon (kind, enabled);
