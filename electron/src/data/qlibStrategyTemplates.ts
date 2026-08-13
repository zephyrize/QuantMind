import {
  strategyTemplates as legacyTemplates,
  type StrategyTemplate as LegacyStrategyTemplate,
} from '../constants/strategyTemplates';

export type TemplateCategory = 'basic' | 'advanced' | 'risk_control';

export interface StrategyTemplateParam {
  name: string;
  type: string;
  default: string | number | boolean;
  description: string;
}

export interface StrategyTemplate {
  id: string;
  name: string;
  category: TemplateCategory;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  tags: string[];
  code: string;
  author?: string;
  params: StrategyTemplateParam[];
}

const categoryFor = (template: LegacyStrategyTemplate): TemplateCategory => {
  if (template.id === 'StopLoss' || template.id === 'VolatilityWeighted' || template.id === 'adaptive_drift') {
    return 'risk_control';
  }
  if (template.difficulty === 'advanced') {
    return 'advanced';
  }
  return 'basic';
};

const normalizeParam = (
  param: NonNullable<LegacyStrategyTemplate['parameters']>[number],
): StrategyTemplateParam => ({
  ...param,
  default:
    typeof param.default === 'string' ||
    typeof param.default === 'number' ||
    typeof param.default === 'boolean'
      ? param.default
      : String(param.default ?? ''),
});

export const QLIB_STRATEGY_TEMPLATES: StrategyTemplate[] = legacyTemplates.map(
  (template) => ({
    id: template.id,
    name: template.name,
    category: categoryFor(template),
    description: template.description,
    difficulty: template.difficulty,
    tags: template.tags,
    code: template.code,
    author: template.author,
    params: (template.parameters || []).map(normalizeParam),
  }),
);

export const getTemplateById = (id: string): StrategyTemplate | undefined =>
  QLIB_STRATEGY_TEMPLATES.find((template) => template.id === id);
