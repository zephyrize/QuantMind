-- OSS runtime schema compatibility migration.
-- Safe to run repeatedly. It upgrades existing databases without dropping data.

BEGIN;

-- The stream archive uses a per-data-source daily aggregate. Older bootstrap
-- SQL created only a legacy (symbol, trade_date) shape.
ALTER TABLE quote_daily_summaries
    ADD COLUMN IF NOT EXISTS avg_price DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volume_sum BIGINT,
    ADD COLUMN IF NOT EXISTS amount_sum DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quote_count INTEGER,
    ADD COLUMN IF NOT EXISTS first_quote_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_quote_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

UPDATE quote_daily_summaries
SET data_source = COALESCE(NULLIF(data_source, ''), 'legacy')
WHERE data_source IS NULL OR data_source = '';

ALTER TABLE quote_daily_summaries
    ALTER COLUMN data_source SET DEFAULT 'remote_redis',
    ALTER COLUMN data_source SET NOT NULL;

ALTER TABLE quote_daily_summaries
    DROP CONSTRAINT IF EXISTS quote_daily_summaries_symbol_trade_date_key;

CREATE UNIQUE INDEX IF NOT EXISTS uq_quote_daily_summaries_trade_date_symbol_source
    ON quote_daily_summaries (trade_date, symbol, data_source);

-- The public API and SQLAlchemy models persist lower-case values. Older
-- bootstrap scripts created upper-case enum labels, causing asyncpg bind
-- failures such as orderstatus = 'submitted'.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_enum e
        JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = 'orderside' AND e.enumlabel = 'BUY'
    ) THEN
        ALTER TYPE orderside RENAME VALUE 'BUY' TO 'buy';
        ALTER TYPE orderside RENAME VALUE 'SELL' TO 'sell';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_enum e
        JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = 'positionside' AND e.enumlabel = 'LONG'
    ) THEN
        ALTER TYPE positionside RENAME VALUE 'LONG' TO 'long';
        ALTER TYPE positionside RENAME VALUE 'SHORT' TO 'short';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_enum e
        JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = 'ordertype' AND e.enumlabel = 'MARKET'
    ) THEN
        ALTER TYPE ordertype RENAME VALUE 'MARKET' TO 'market';
        ALTER TYPE ordertype RENAME VALUE 'LIMIT' TO 'limit';
        ALTER TYPE ordertype RENAME VALUE 'STOP' TO 'stop';
        ALTER TYPE ordertype RENAME VALUE 'STOP_LIMIT' TO 'stop_limit';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_enum e
        JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = 'orderstatus' AND e.enumlabel = 'PENDING'
    ) THEN
        ALTER TYPE orderstatus RENAME VALUE 'PENDING' TO 'pending';
        ALTER TYPE orderstatus RENAME VALUE 'SUBMITTED' TO 'submitted';
        ALTER TYPE orderstatus RENAME VALUE 'PARTIAL_FILL' TO 'partially_filled';
        ALTER TYPE orderstatus RENAME VALUE 'FILLED' TO 'filled';
        ALTER TYPE orderstatus RENAME VALUE 'CANCELLED' TO 'cancelled';
        ALTER TYPE orderstatus RENAME VALUE 'REJECTED' TO 'rejected';
        ALTER TYPE orderstatus RENAME VALUE 'EXPIRED' TO 'expired';
    END IF;
END $$;

-- Legacy trade_action labels do not describe buy/sell direction. Rebuild this
-- enum once and derive the new value from existing side + position_side,
-- preserving execution semantics for historical rows.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_enum e
        JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = 'tradeaction' AND e.enumlabel = 'OPEN'
    ) THEN
        EXECUTE 'ALTER TYPE tradeaction RENAME TO tradeaction_legacy';
        EXECUTE $sql$
            CREATE TYPE tradeaction AS ENUM (
                'buy_to_open',
                'sell_to_close',
                'sell_to_open',
                'buy_to_close'
            )
        $sql$;
        EXECUTE $sql$
            ALTER TABLE orders
            ALTER COLUMN trade_action TYPE tradeaction
            USING (
                CASE
                    WHEN trade_action IS NULL THEN NULL
                    WHEN side::text = 'buy' AND position_side::text = 'long'
                        THEN 'buy_to_open'
                    WHEN side::text = 'sell' AND position_side::text = 'long'
                        THEN 'sell_to_close'
                    WHEN side::text = 'sell' AND position_side::text = 'short'
                        THEN 'sell_to_open'
                    WHEN side::text = 'buy' AND position_side::text = 'short'
                        THEN 'buy_to_close'
                    ELSE NULL
                END
            )::tradeaction
        $sql$;
        EXECUTE $sql$
            ALTER TABLE trades
            ALTER COLUMN trade_action TYPE tradeaction
            USING (
                CASE
                    WHEN trade_action IS NULL THEN NULL
                    WHEN side::text = 'buy' AND position_side::text = 'long'
                        THEN 'buy_to_open'
                    WHEN side::text = 'sell' AND position_side::text = 'long'
                        THEN 'sell_to_close'
                    WHEN side::text = 'sell' AND position_side::text = 'short'
                        THEN 'sell_to_open'
                    WHEN side::text = 'buy' AND position_side::text = 'short'
                        THEN 'buy_to_close'
                    ELSE NULL
                END
            )::tradeaction
        $sql$;
        EXECUTE 'DROP TYPE tradeaction_legacy';
    END IF;
END $$;

COMMIT;
