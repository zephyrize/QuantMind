-- 2026-05-25 实体级情感: JSONB { "ticker:600519.SH": 0.7, "country:美国": -0.4, ... }
ALTER TABLE news_article_enrichment
  ADD COLUMN IF NOT EXISTS entity_sentiments jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_enr_entity_sentiments ON news_article_enrichment USING GIN (entity_sentiments);

SELECT '实体情感列添加完成' AS msg;
