-- 2026-05-25 enrichment 扩列: 中国国内地理 + 领导人 + 调研
ALTER TABLE news_article_enrichment
  ADD COLUMN IF NOT EXISTS provinces   text[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS cities      text[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS politicians text[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS visits      text[] NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_enr_provinces   ON news_article_enrichment USING GIN (provinces);
CREATE INDEX IF NOT EXISTS idx_enr_cities      ON news_article_enrichment USING GIN (cities);
CREATE INDEX IF NOT EXISTS idx_enr_politicians ON news_article_enrichment USING GIN (politicians);
CREATE INDEX IF NOT EXISTS idx_enr_visits      ON news_article_enrichment USING GIN (visits);

SELECT 'enrichment 中国地理扩列完成' AS msg;
