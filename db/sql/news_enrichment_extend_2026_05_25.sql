-- 2026-05-25 enrichment 扩列：countries / regions / key_terms / dates
ALTER TABLE news_article_enrichment
  ADD COLUMN IF NOT EXISTS countries     text[]  NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS regions       text[]  NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS key_terms     text[]  NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS date_entities text[]  NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_enr_countries     ON news_article_enrichment USING GIN (countries);
CREATE INDEX IF NOT EXISTS idx_enr_regions       ON news_article_enrichment USING GIN (regions);
CREATE INDEX IF NOT EXISTS idx_enr_key_terms     ON news_article_enrichment USING GIN (key_terms);
CREATE INDEX IF NOT EXISTS idx_enr_date_entities ON news_article_enrichment USING GIN (date_entities);

SELECT 'enrichment 扩列完成' AS msg;
