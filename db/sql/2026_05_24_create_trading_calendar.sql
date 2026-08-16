-- 多市场交易日历
-- 由 D8 cron 每日从主备数据源同步（A: baostock; HK: efinance; US: yahoo_finance）
-- D2 引入；适配器 ChinaACalendar/HongKongCalendar/UnitedStatesCalendar 优先查询此表。

CREATE TABLE IF NOT EXISTS trading_calendar (
    market         VARCHAR(8)  NOT NULL,        -- A / HK / US
    trade_date     DATE        NOT NULL,
    is_trading     BOOLEAN     NOT NULL,
    is_half_day    BOOLEAN     NOT NULL DEFAULT FALSE,
    note           TEXT,                        -- 节假日名称等
    source         VARCHAR(32),                 -- 写入来源 (baostock/efinance/yahoo_finance/manual)
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (market, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_trading_calendar_market_trading
    ON trading_calendar (market, is_trading, trade_date);
