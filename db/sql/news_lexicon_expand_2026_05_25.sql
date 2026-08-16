-- 2026-05-25 扩充财经词典：
-- 1) 市场板块 (market): A股/港股/美股/外汇/期货/原油/区块链/比特币/黄金 等
-- 2) 财务事件: 财报/业绩说明会/盈利预警/营收 等
-- 3) 强情感词 (强利好/强利空)

INSERT INTO finance_lexicon (term, kind, event_tag, weight, note) VALUES
  -- ---------- 市场板块 (event_tag='市场') ----------
  ('A股',        'event', '市场',  1.0, '市场板块: A股'),
  ('沪市',       'event', '市场',  1.0, ''),
  ('深市',       'event', '市场',  1.0, ''),
  ('沪深',       'event', '市场',  1.0, ''),
  ('创业板',     'event', '市场',  1.0, ''),
  ('科创板',     'event', '市场',  1.0, ''),
  ('北交所',     'event', '市场',  1.0, ''),
  ('港股',       'event', '市场',  1.0, '香港市场'),
  ('恒生指数',   'event', '市场',  1.0, ''),
  ('恒指',       'event', '市场',  1.0, ''),
  ('美股',       'event', '市场',  1.0, '美国市场'),
  ('纳斯达克',   'event', '市场',  1.0, ''),
  ('标普500',    'event', '市场',  1.0, ''),
  ('道琼斯',     'event', '市场',  1.0, ''),
  ('中概股',     'event', '市场',  1.0, '中概股回归/退市相关'),
  ('A50',        'event', '市场',  1.0, '富时中国A50'),

  -- ---------- 大宗 / 期货 (event_tag='期货') ----------
  ('期货',       'event', '期货',  1.0, ''),
  ('原油',       'event', '期货',  1.0, '原油价格'),
  ('布伦特',     'event', '期货',  1.0, '布伦特原油'),
  ('WTI',        'event', '期货',  1.0, '西德州原油'),
  ('天然气',     'event', '期货',  1.0, ''),
  ('黄金',       'event', '期货',  1.0, ''),
  ('白银',       'event', '期货',  1.0, ''),
  ('铜价',       'event', '期货',  1.0, ''),
  ('铁矿石',     'event', '期货',  1.0, ''),
  ('煤炭',       'event', '期货',  1.0, ''),
  ('焦炭',       'event', '期货',  1.0, ''),
  ('动力煤',     'event', '期货',  1.0, ''),
  ('豆粕',       'event', '期货',  1.0, ''),
  ('生猪',       'event', '期货',  1.0, ''),
  ('棉花',       'event', '期货',  1.0, ''),

  -- ---------- 外汇 / 货币 (event_tag='外汇') ----------
  ('汇率',       'event', '外汇',  1.0, ''),
  ('人民币',     'event', '外汇',  1.0, ''),
  ('美元指数',   'event', '外汇',  1.0, ''),
  ('美元兑人民币','event', '外汇', 1.0, ''),
  ('离岸人民币', 'event', '外汇',  1.0, ''),
  ('在岸人民币', 'event', '外汇',  1.0, ''),
  ('日元',       'event', '外汇',  1.0, ''),
  ('欧元',       'event', '外汇',  1.0, ''),
  ('英镑',       'event', '外汇',  1.0, ''),
  ('港币',       'event', '外汇',  1.0, ''),

  -- ---------- 加密 (event_tag='加密') ----------
  ('区块链',     'event', '加密',  1.0, ''),
  ('比特币',     'event', '加密',  1.0, ''),
  ('以太坊',     'event', '加密',  1.0, ''),
  ('以太币',     'event', '加密',  1.0, ''),
  ('BTC',        'event', '加密',  1.0, ''),
  ('ETH',        'event', '加密',  1.0, ''),
  ('稳定币',     'event', '加密',  1.0, ''),
  ('数字货币',   'event', '加密',  1.0, ''),
  ('加密货币',   'event', '加密',  1.0, ''),
  ('NFT',        'event', '加密',  1.0, ''),
  ('Web3',       'event', '加密',  1.0, ''),

  -- ---------- 宏观 / 央行 (event_tag='宏观') ----------
  ('美联储',     'event', '宏观',  1.0, ''),
  ('央行',       'event', '宏观',  1.0, ''),
  ('人民银行',   'event', '宏观',  1.0, ''),
  ('LPR',        'event', '宏观',  1.0, '贷款基础利率'),
  ('MLF',        'event', '宏观',  1.0, '中期借贷便利'),
  ('降准',       'event', '宏观',  1.0, ''),
  ('降息',       'event', '宏观',  1.0, ''),
  ('加息',       'event', '宏观',  1.0, ''),
  ('CPI',        'event', '宏观',  1.0, ''),
  ('PPI',        'event', '宏观',  1.0, ''),
  ('PMI',        'event', '宏观',  1.0, ''),
  ('GDP',        'event', '宏观',  1.0, ''),
  ('社融',       'event', '宏观',  1.0, '社会融资'),
  ('M2',         'event', '宏观',  1.0, ''),
  ('财政刺激',   'event', '宏观',  1.0, ''),
  ('特别国债',   'event', '宏观',  1.0, ''),
  ('地方债',     'event', '宏观',  1.0, ''),

  -- ---------- 财报 (event_tag='财报', 复用/补充) ----------
  ('财报',          'event', '财报',  1.0, ''),
  ('年报',          'event', '财报',  1.0, ''),
  ('一季报',        'event', '财报',  1.0, ''),
  ('半年报',        'event', '财报',  1.0, ''),
  ('三季报',        'event', '财报',  1.0, ''),
  ('业绩说明会',    'event', '财报',  1.0, ''),
  ('营业收入',      'event', '财报',  1.0, ''),
  ('净利润',        'event', '财报',  1.0, ''),
  ('归母净利润',    'event', '财报',  1.0, ''),
  ('扣非净利润',    'event', '财报',  1.0, ''),
  ('毛利率',        'event', '财报',  1.0, ''),
  ('盈利预警',      'event', '财报',  1.0, '港股常见'),
  ('盈喜',          'event', '财报',  1.0, '港股利好预告'),

  -- ---------- 政策 / 监管 ----------
  ('证监会',     'event', '监管',  1.0, ''),
  ('交易所',     'event', '监管',  1.0, ''),
  ('国务院',     'event', '监管',  1.0, ''),
  ('发改委',     'event', '监管',  1.0, ''),
  ('工信部',     'event', '监管',  1.0, '')

ON CONFLICT (term, kind) DO UPDATE
SET event_tag = EXCLUDED.event_tag,
    weight    = EXCLUDED.weight,
    note      = COALESCE(EXCLUDED.note, finance_lexicon.note),
    enabled   = TRUE;


-- 强情感词 (新词或加权升级)
INSERT INTO finance_lexicon (term, kind, event_tag, weight, note) VALUES
  -- 强利好 weight=2.0
  ('大涨',       'sentiment_pos', NULL, 2.0, ''),
  ('暴涨',       'sentiment_pos', NULL, 2.5, ''),
  ('飙升',       'sentiment_pos', NULL, 2.0, ''),
  ('涨停',       'sentiment_pos', NULL, 2.0, ''),
  ('连板',       'sentiment_pos', NULL, 2.0, ''),
  ('利好',       'sentiment_pos', NULL, 1.5, ''),
  ('重磅利好',   'sentiment_pos', NULL, 2.5, ''),
  ('超预期',     'sentiment_pos', NULL, 1.8, ''),
  ('创新高',     'sentiment_pos', NULL, 1.5, ''),
  ('破纪录',     'sentiment_pos', NULL, 1.5, ''),
  ('翻倍',       'sentiment_pos', NULL, 1.8, ''),
  ('井喷',       'sentiment_pos', NULL, 1.8, ''),
  ('放量',       'sentiment_pos', NULL, 1.0, ''),
  ('反弹',       'sentiment_pos', NULL, 1.0, ''),
  ('扭亏',       'sentiment_pos', NULL, 1.8, ''),
  ('扭亏为盈',   'sentiment_pos', NULL, 2.0, ''),
  ('大幅增长',   'sentiment_pos', NULL, 1.8, ''),
  ('强劲',       'sentiment_pos', NULL, 1.2, ''),

  -- 强利空 weight=2.0
  ('大跌',       'sentiment_neg', NULL, 2.0, ''),
  ('暴跌',       'sentiment_neg', NULL, 2.5, ''),
  ('崩盘',       'sentiment_neg', NULL, 3.0, ''),
  ('跌停',       'sentiment_neg', NULL, 2.0, ''),
  ('利空',       'sentiment_neg', NULL, 1.5, ''),
  ('重大利空',   'sentiment_neg', NULL, 2.5, ''),
  ('低于预期',   'sentiment_neg', NULL, 1.8, ''),
  ('不及预期',   'sentiment_neg', NULL, 1.8, ''),
  ('爆雷',       'sentiment_neg', NULL, 2.5, ''),
  ('暴雷',       'sentiment_neg', NULL, 2.5, ''),
  ('巨亏',       'sentiment_neg', NULL, 2.0, ''),
  ('破发',       'sentiment_neg', NULL, 1.5, ''),
  ('跳水',       'sentiment_neg', NULL, 2.0, ''),
  ('腰斩',       'sentiment_neg', NULL, 2.5, ''),
  ('退市风险',   'sentiment_neg', NULL, 2.0, ''),
  ('ST警示',     'sentiment_neg', NULL, 1.8, ''),
  ('停产',       'sentiment_neg', NULL, 1.5, ''),
  ('停摆',       'sentiment_neg', NULL, 1.5, ''),
  ('清盘',       'sentiment_neg', NULL, 2.5, ''),
  ('违约',       'sentiment_neg', NULL, 1.8, ''),
  ('债务危机',   'sentiment_neg', NULL, 2.5, ''),
  ('立案调查',   'sentiment_neg', NULL, 2.0, ''),
  ('被处罚',     'sentiment_neg', NULL, 1.5, ''),
  ('监管处罚',   'sentiment_neg', NULL, 1.5, ''),
  ('财务造假',   'sentiment_neg', NULL, 3.0, '')

ON CONFLICT (term, kind) DO UPDATE
SET weight = GREATEST(EXCLUDED.weight, finance_lexicon.weight),
    enabled = TRUE;

SELECT kind, COUNT(*) FROM finance_lexicon WHERE enabled GROUP BY kind ORDER BY 2 DESC;
