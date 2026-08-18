import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Layers, Star, RefreshCw, Search, Code, Calendar, Layers2,
  History, Archive, Brain, Clock, XCircle, X, Trash2,
  ChevronRight, Play, Cpu, Download, ChevronDown,
  ChevronUp, Shield, Zap, Activity, ListFilter, BarChart3, TrendingUp, TrendingDown,
} from 'lucide-react';
import {
  Button, Card, Tag, Typography, Empty, Spin, message,
  Progress, Divider, Row, Col, Input, Modal, Tabs, Switch,
  DatePicker, Table, Badge, Tooltip, Select,
} from 'antd';
import { clsx } from 'clsx';
import dayjs from 'dayjs';
import { useNavigate } from 'react-router-dom';
import {
  modelTrainingService,
  UserModelRecord,
  SystemModelRecord,
  ModelTrainingRunStatus,
  InferenceRunRecord,
  InferencePrecheckResult,
  AutoInferenceSettings,
  LatestInferenceRunInfo,
  ModelShapSummaryResponse,
} from '../services/modelTrainingService';
import {
  calcTimeSplitStats,
  extractModelType,
  extractTimePeriods,
  formatTrendLabel,
  getMeta,
  getMetrics,
  getStatusConfig,
  isSystemModel,
  modelDisplayName,
  resolveMetricNumber,
  systemModelToUserModel,
} from './modelRegistryUtils';
import {
  ModelCard,
  ModelDetailPanel,
  TrainingSourcePanel,
  AttributionAnalysisPanel,
  InferenceCenterPanel,
  MetricCard,
  TimeItem,
  InfoCell,
} from './modelRegistryPanels';
import { CreateEnsembleModal } from './CreateEnsembleModal';
import { InferenceBacktestModule } from '../components/backtestCenter/InferenceBacktestModule';
import { BatchInferencePanel } from '../components/inference/BatchInferencePanel';
import { BatchSingleDayPanel } from '../components/inference/BatchSingleDayPanel';
import { InferenceHistoryPanel } from '../components/inference/InferenceHistoryPanel';
import { ModelScoreResearch } from '../components/inference/ModelScoreResearch';
import { StockPickingPanel } from '../components/inference/StockPickingPanel';
import { NegativeScorePanel } from '../components/inference/NegativeScorePanel';
import {
  buildFeatureLabelMap,
  DEFAULT_FEATURE_CATEGORIES,
  toDynamicCategories,
} from './training/trainingUtils';
import { PAGE_LAYOUT } from '../config/pageLayout';
import { useAppSelector } from '../store';
import { selectCurrentMarket } from '../store/slices/uiSlice';
import { getMarketConfig } from '../config/marketConfig';
const { Text } = Typography;

export const ModelRegistryPage: React.FC = () => {
  const navigate = useNavigate();
  const currentMarket = useAppSelector(selectCurrentMarket);
  const marketConfig = getMarketConfig(currentMarket);
  const [loading, setLoading] = useState(true);
  const [userModels, setUserModels] = useState<UserModelRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [ensembleMode, setEnsembleMode] = useState(false);
  const [ensembleChecked, setEnsembleChecked] = useState<string[]>([]);
  const [showEnsembleModal, setShowEnsembleModal] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [settingDefault, setSettingDefault] = useState(false);
  const [archiving, setArchiving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [mainTab, setMainTab] = useState('detail');
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [activeConfigTab, setActiveConfigTab] = useState<'meta' | 'metrics'>('meta');
  const [trainingRun, setTrainingRun] = useState<ModelTrainingRunStatus | null>(null);
  const [trainingRunLoading, setTrainingRunLoading] = useState(false);
  const [shapSummary, setShapSummary] = useState<ModelShapSummaryResponse | null>(null);
  const [shapLoading, setShapLoading] = useState(false);
  const [shapError, setShapError] = useState('');
  const [featureLabelMap, setFeatureLabelMap] = useState<Record<string, string>>(() => buildFeatureLabelMap(DEFAULT_FEATURE_CATEGORIES));
  const [featureCatalogLoaded, setFeatureCatalogLoaded] = useState(false);
  const [inferenceDate, setInferenceDate] = useState<dayjs.Dayjs | null>(dayjs());
  const [inferenceRunning, setInferenceRunning] = useState(false);
  const [inferenceMode, setInferenceMode] = useState<'single' | 'batch' | 'batch-range' | 'history'>('single');
  const [lastInferenceRun, setLastInferenceRun] = useState<InferenceRunRecord | null>(null);
  const [inferenceHistory, setInferenceHistory] = useState<InferenceRunRecord[]>([]);
  const [inferenceHistoryLoading, setInferenceHistoryLoading] = useState(false);
  const [inferencePrecheck, setInferencePrecheck] = useState<InferencePrecheckResult | null>(null);
  const [inferencePrecheckLoading, setInferencePrecheckLoading] = useState(false);
  const [inferenceTargetDate, setInferenceTargetDate] = useState<string>('—');
  const [inferenceTargetLoading, setInferenceTargetLoading] = useState(false);
  const [autoSettings, setAutoSettings] = useState<AutoInferenceSettings | null>(null);
  const [autoSaving, setAutoSaving] = useState(false);
  const [latestInferenceRun, setLatestInferenceRun] = useState<LatestInferenceRunInfo | null>(null);
  const [latestInferenceRunLoading, setLatestInferenceRunLoading] = useState(false);
  const [historyRunIdFilter, setHistoryRunIdFilter] = useState('');
  const [historyStatusFilter, setHistoryStatusFilter] = useState<'all' | 'running' | 'completed' | 'failed'>('all');
  const [historyDateFilter, setHistoryDateFilter] = useState<dayjs.Dayjs | null>(null);

  const allModels = userModels;
  const activeModels = allModels.filter(m => m.status !== 'archived');
  const archivedModels = allModels.filter(m => m.status === 'archived');
  const displayModels = (showArchived ? allModels : activeModels).filter(m =>
    !searchQuery ||
    m.model_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    modelDisplayName(m).toLowerCase().includes(searchQuery.toLowerCase())
  );

  const selectedModel = userModels.find(m => m.model_id === selectedId) ?? null;
  const meta = selectedModel ? getMeta(selectedModel) : {} as ReturnType<typeof getMeta>;
  const metrics = selectedModel ? getMetrics(selectedModel) : {} as ReturnType<typeof getMetrics>;
  const timePeriods = selectedModel ? extractTimePeriods(getMeta(selectedModel)) : null;
  const horizonDays = Number(meta?.target_horizon_days ?? meta?.horizon_days ?? 3);

  const loadModels = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const [resp, sysModels] = await Promise.all([
        modelTrainingService.listUserModels(true),
        modelTrainingService.listSystemModels(),
      ]);
      const allItems = resp.items ?? [];
      // 按当前市场过滤用户模型：老模型（无 market 字段）仅在 CN 市场显示
      const items = allItems.filter((m) => {
        const meta = m.metadata_json || {};
        const modelMarket = ((meta.market as string) || '').toUpperCase();
        if (!modelMarket) return currentMarket === 'CN';
        return modelMarket === currentMarket;
      });
      // 系统模型转为 UserModelRecord 格式并合并到列表顶部
      const sysItems: UserModelRecord[] = (sysModels ?? [])
        .filter((sm) => {
          const meta = sm as unknown as Record<string, unknown>;
          const mkt = ((meta.market as string) || '').toUpperCase();
          if (!mkt) return currentMarket === 'CN';
          return mkt === currentMarket;
        })
        .map((sm) => ({
          tenant_id: 'system',
          user_id: 'system',
          model_id: sm.model_id,
          source_run_id: '',
          status: 'active',
          storage_path: '',
          model_file: '',
          metadata_json: {
            display_name: sm.display_name,
            description: sm.description,
            framework: sm.framework,
            model_type: sm.model_type,
            feature_count: sm.feature_count,
            feature_columns: sm.feature_columns,
            market: (sm as unknown as Record<string, unknown>).market,
            ensemble_config: (sm as unknown as Record<string, unknown>).ensemble_config,
          },
          metrics_json: sm.performance_metrics ?? {},
          is_default: false,
          created_at: sm.created_at,
        }));
      const merged = [...sysItems, ...items];
      setUserModels(merged);

      if (!selectedId && merged.length > 0) {
        const def = merged.find(m => m.is_default) ?? merged[0];
        if (def) setSelectedId(def.model_id);
      }
    } catch (err: any) {
      message.error(`加载模型列表失败: ${err?.message ?? '未知错误'}`);
    } finally {
      setLoading(false);
    }
  }, [selectedId, currentMarket]);

  useEffect(() => { loadModels(); }, []);
  useEffect(() => { loadModels(); }, [currentMarket]);

  useEffect(() => {
    setMainTab('detail');
    setTrainingRun(null);
    setShapSummary(null);
    setShapLoading(false);
    setShapError('');
    setLastInferenceRun(null);
    setInferenceHistory([]);
    setInferencePrecheck(null);
    setAutoSettings(null);
    setLatestInferenceRun(null);
    setHistoryRunIdFilter('');
    setHistoryStatusFilter('all');
    setHistoryDateFilter(null);
    setInferenceTargetDate('—');
    setInferenceTargetLoading(false);
  }, [selectedId]);

  const handleTabChange = (key: string) => {
    setMainTab(key);
    if (key === 'training' && selectedModel?.source_run_id && !trainingRun) {
      loadTrainingRun(selectedModel.source_run_id);
    }
    if (key === 'attribution' && selectedModel && !shapSummary && !shapLoading) {
      void loadFeatureLabelCatalog();
      void loadShapSummary(selectedModel.model_id);
    }
    if (key === 'inference' && selectedModel) {
      void refreshInferencePanel(selectedModel.model_id);
    }
  };

  const loadTrainingRun = useCallback(async (runId: string) => {
    setTrainingRunLoading(true);
    try {
      const run = await modelTrainingService.getTrainingRun(runId);
      setTrainingRun(run);
    } catch { setTrainingRun(null); }
    finally { setTrainingRunLoading(false); }
  }, []);

  const loadShapSummary = useCallback(async (modelId: string) => {
    setShapLoading(true);
    setShapError('');
    try {
      const summary = await modelTrainingService.getModelShapSummary(modelId);
      setShapSummary(summary);
    } catch (err: any) {
      setShapSummary(null);
      const detail = err?.response?.data?.detail;
      setShapError(String(detail || err?.message || '加载 SHAP 归因结果失败'));
    } finally {
      setShapLoading(false);
    }
  }, []);

  const loadFeatureLabelCatalog = useCallback(async () => {
    if (featureCatalogLoaded) return;
    try {
      const catalog = await modelTrainingService.getFeatureCatalog();
      const categories = toDynamicCategories(catalog);
      if (categories.length > 0) {
        setFeatureLabelMap(buildFeatureLabelMap(categories));
      }
      setFeatureCatalogLoaded(true);
    } catch {
      setFeatureCatalogLoaded(true);
    }
  }, [featureCatalogLoaded]);

  const loadPrecheck = useCallback(async (modelId: string, inferenceDate?: string) => {
    setInferencePrecheckLoading(true);
    try {
      const resp = await modelTrainingService.precheckInference(modelId, inferenceDate);
      setInferencePrecheck(resp);
      if (resp?.prediction_trade_date) {
        setInferenceTargetDate(resp.prediction_trade_date);
      }
      return resp;
    } catch {
      setInferencePrecheck(null);
      return null;
    } finally {
      setInferencePrecheckLoading(false);
    }
  }, []);

  const loadInferenceTargetDate = useCallback(async () => {
    if (!inferenceDate) {
      setInferenceTargetDate('—');
      return;
    }
    setInferenceTargetLoading(true);
    try {
      const base = inferenceDate.format('YYYY-MM-DD');
      const resolved = await modelTrainingService.resolveInferenceDateByCalendar(marketConfig.calendar, base);
      const predicted = await modelTrainingService.calcTargetDateByCalendar(marketConfig.calendar, resolved.date, horizonDays);
      setInferenceTargetDate(predicted || '—');
    } catch {
      setInferenceTargetDate('—');
    } finally {
      setInferenceTargetLoading(false);
    }
  }, [inferenceDate, horizonDays]);

  const loadInferenceHistory = useCallback(async (
    modelId: string,
    options?: {
      runId?: string;
      status?: string;
      inferenceDate?: string;
      page?: number;
      pageSize?: number;
    },
  ) => {
    setInferenceHistoryLoading(true);
    try {
      const resp = await modelTrainingService.listInferenceHistory(modelId, {
        runId: options?.runId,
        status: options?.status,
        inferenceDate: options?.inferenceDate,
        page: options?.page ?? 1,
        pageSize: options?.pageSize ?? 20,
      });
      setInferenceHistory(resp.items);
    } catch { setInferenceHistory([]); }
    finally { setInferenceHistoryLoading(false); }
  }, []);

  const loadAutoSettings = useCallback(async (modelId: string) => {
    try {
      const s = await modelTrainingService.getAutoInferenceSettings(modelId);
      setAutoSettings(s);
    } catch { setAutoSettings(null); }
  }, []);

  const loadLatestInferenceRun = useCallback(async (modelId: string) => {
    setLatestInferenceRunLoading(true);
    try {
      const latest = await modelTrainingService.getLatestInferenceRun(modelId);
      setLatestInferenceRun(latest);
    } catch {
      setLatestInferenceRun(null);
    } finally {
      setLatestInferenceRunLoading(false);
    }
  }, []);

  const refreshInferencePanel = useCallback(async (modelId: string) => {
    const currentDate = inferenceDate ? inferenceDate.format('YYYY-MM-DD') : undefined;
    await Promise.all([
      loadPrecheck(modelId, currentDate),
      loadAutoSettings(modelId),
      loadLatestInferenceRun(modelId),
    ]);
  }, [inferenceDate, loadAutoSettings, loadLatestInferenceRun, loadPrecheck]);

  useEffect(() => {
    if (selectedModel && mainTab === 'inference') {
      void refreshInferencePanel(selectedModel.model_id);
    }
  }, [selectedModel?.model_id, mainTab, inferenceDate, refreshInferencePanel]);

  useEffect(() => {
    if (mainTab === 'inference') {
      void loadInferenceTargetDate();
    }
  }, [mainTab, loadInferenceTargetDate]);

  useEffect(() => {
    if (selectedModel && mainTab === 'inference') {
      void loadInferenceHistory(selectedModel.model_id, {
        runId: historyRunIdFilter || undefined,
        status: historyStatusFilter === 'all' ? undefined : historyStatusFilter,
        inferenceDate: historyDateFilter ? historyDateFilter.format('YYYY-MM-DD') : undefined,
        page: 1,
        pageSize: 20,
      });
    }
  }, [selectedModel?.model_id, mainTab, historyRunIdFilter, historyStatusFilter, historyDateFilter, loadInferenceHistory]);

  const handleSetDefault = async () => {
    if (!selectedModel) return;
    await handleSetDefaultById(selectedModel.model_id);
  };

  const handleSetDefaultById = async (modelId: string) => {
    setSettingDefault(true);
    const canonicalId = modelId.startsWith('sys-') ? modelId.slice(4) : modelId;
    try {
      await modelTrainingService.setDefaultModel(canonicalId);
      message.success(`已设为默认模型：${modelId}`);
      await loadModels(true);
      setSelectedId(modelId);
    } catch (err: any) {
      message.error(`设置失败: ${err?.message ?? '未知'}`);
    } finally { setSettingDefault(false); }
  };

  const handleArchive = () => {
    if (!selectedModel) return;
    Modal.confirm({
      title: '归档模型',
      content: `确定归档 "${selectedModel.model_id}"？归档后不再参与推理，但数据不会删除。`,
      okText: '确认归档', okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        setArchiving(true);
        try {
          await modelTrainingService.archiveUserModel(selectedModel.model_id);
          message.success('已归档');
          setSelectedId(null);
          await loadModels(true);
        } catch (err: any) {
          message.error(`归档失败: ${err?.message ?? '未知'}`);
        } finally { setArchiving(false); }
      },
    });
  };

  const handleDeleteArchived = () => {
    if (!selectedModel || selectedModel.status !== 'archived') return;
    Modal.confirm({
      title: '永久删除模型',
      content: `确定永久删除 "${selectedModel.model_id}" 吗？模型文件、元数据、策略绑定和自动推理设置将被删除，且无法恢复。`,
      okText: '永久删除', okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        setDeleting(true);
        try {
          const result = await modelTrainingService.deleteArchivedUserModel(selectedModel.model_id);
          if (result.artifacts_deleted) {
            message.success('模型及其文件已永久删除');
          } else {
            message.warning('模型记录已删除，但模型文件清理未完成，请联系管理员处理');
          }
          setSelectedId(null);
          await loadModels(true);
        } catch (err: any) {
          const detail = err?.response?.data?.detail ?? err?.message ?? '未知错误';
          message.error(`删除失败: ${detail}`);
        } finally { setDeleting(false); }
      },
    });
  };

  const handleRunInference = async () => {
    if (!selectedModel || !inferenceDate) return;
    setInferenceRunning(true);
    setLastInferenceRun(null);
    try {
      const requestedDateStr = inferenceDate.format('YYYY-MM-DD');
      const resolvedDate = await modelTrainingService.resolveInferenceDateByCalendar(marketConfig.calendar, requestedDateStr);
      const inferenceDateStr = resolvedDate.date;
      if (resolvedDate.adjusted && inferenceDateStr) {
        setInferenceDate(dayjs(inferenceDateStr));
        message.info(`所选日期 ${requestedDateStr} 非交易日，已自动回退到最近交易日 ${inferenceDateStr}`);
      }
      const precheck = await loadPrecheck(selectedModel.model_id, inferenceDateStr);
      if (!precheck?.passed) {
        message.error('前置检查未通过，请先处理阻断项');
        return;
      }
      const run = await modelTrainingService.runModelInference(
        selectedModel.model_id,
        inferenceDateStr,
      );
      setLastInferenceRun(run);
      if (run.success) {
        message.success(`推理完成，共生成 ${run.signals_count} 支排名信号`);
      } else {
        message.warning(run.error_message || run.fallback_reason || '推理执行完成但返回失败状态');
      }
      await refreshInferencePanel(selectedModel.model_id);
      await loadInferenceHistory(selectedModel.model_id, {
        runId: historyRunIdFilter || undefined,
        status: historyStatusFilter === 'all' ? undefined : historyStatusFilter,
        inferenceDate: historyDateFilter ? historyDateFilter.format('YYYY-MM-DD') : undefined,
        page: 1,
        pageSize: 20,
      });
    } catch (err: any) {
      message.error(`推理失败: ${err?.message ?? '未知'}`);
    } finally { setInferenceRunning(false); }
  };

  const handleToggleAuto = async (enabled: boolean) => {
    if (!selectedModel || !autoSettings) return;
    setAutoSaving(true);
    try {
      const next = { ...autoSettings, enabled };
      const saved = await modelTrainingService.saveAutoInferenceSettings(selectedModel.model_id, next);
      setAutoSettings(saved);
      message.success(enabled ? '自动推理已开启' : '自动推理已关闭');
    } catch { message.error('保存失败'); }
    finally { setAutoSaving(false); }
  };

  const handleDeleteHistory = (runId: string) => {
    Modal.confirm({
      title: '删除推理历史',
      content: `确定要删除推理批次 "${runId}" 吗？此操作将同步删除关联的预测信号，且不可撤销。`,
      okText: '确定删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        try {
          await modelTrainingService.deleteInferenceHistory(runId);
          message.success('推理历史已删除');
          if (selectedModel) {
            void loadInferenceHistory(selectedModel.model_id, {
              runId: historyRunIdFilter || undefined,
              status: historyStatusFilter === 'all' ? undefined : historyStatusFilter,
              inferenceDate: historyDateFilter ? historyDateFilter.format('YYYY-MM-DD') : undefined,
              page: 1,
              pageSize: 20,
            });
          }
        } catch (err: any) {
          message.error(`删除失败: ${err?.message ?? '未知错误'}`);
        }
      },
    });
  };

  const targetDate = inferenceTargetDate || '—';

  return (
    <div className={PAGE_LAYOUT.outerClass}>
      <div className={PAGE_LAYOUT.frameClass}>
        <div className="flex flex-row h-full w-full overflow-hidden">
          {/* ═══ 左侧边栏 ═══ */}
          <div className="w-[300px] flex-shrink-0 border-r border-slate-100 bg-white flex flex-col shadow-lg shadow-slate-100/50 z-10 h-full overflow-hidden">
            {/* 顶部标题 + 搜索 */}
            <div className="px-5 pt-5 pb-4 border-b border-slate-50">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-blue-600 flex items-center justify-center shadow shadow-blue-500/30 text-white">
                    <Layers size={17} />
                  </div>
                  <span className="text-[15px] font-black text-slate-800 tracking-tight">模型资产库 ({marketConfig.label})</span>
                </div>
                <button
                  onClick={() => loadModels()}
                  className="w-7 h-7 flex items-center justify-center rounded-lg text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition-all"
                >
                  <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
                </button>
              </div>
              <Input
                prefix={<Search size={13} className="text-slate-300" />}
                placeholder="搜索模型..."
                className="rounded-xl border-slate-100 bg-slate-50 h-9 text-xs"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
            </div>
            {/* 分类切换 */}
            <div className="px-4 pt-3 pb-2 flex gap-1">
              <button
                onClick={() => setShowArchived(false)}
                className={clsx(
                  'flex-1 py-1 rounded-lg text-[10px] font-black tracking-widest transition-all',
                  !showArchived ? 'bg-blue-600 text-white shadow shadow-blue-200' : 'text-slate-400 hover:text-slate-600'
                )}
              >使用中 ({activeModels.length})</button>
              <button
                onClick={() => setShowArchived(true)}
                className={clsx(
                  'flex-1 py-1 rounded-lg text-[10px] font-black tracking-widest transition-all',
                  showArchived ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-600'
                )}
              >已归档 ({archivedModels.length})</button>
            </div>
            {/* 模型列表 */}
            <div className="flex-1 overflow-y-auto px-3 pb-5 space-y-1.5 custom-scrollbar">
              {loading ? (
                <div className="flex items-center justify-center py-16"><Spin /></div>
              ) : displayModels.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 gap-4">
                  <Empty description={<span className="text-xs text-slate-400">暂无模型</span>} />
                  <Button
                    type="primary" size="small"
                    className="rounded-xl bg-blue-600 border-none font-bold text-xs"
                    icon={<Brain size={12} />}
                    onClick={() => navigate('/model-training')}
                  >去训练模型</Button>
                </div>
              ) : (
                <>
                  <div className="px-2 pt-2 pb-1 flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <Brain size={10} className="text-blue-500" />
                      <span className="text-[9px] font-black text-blue-500 tracking-widest">我的模型资产</span>
                    </div>
                    {!ensembleMode ? (
                      <button
                        className="text-[9px] font-bold text-blue-600 hover:text-blue-700 flex items-center gap-0.5 bg-blue-50 hover:bg-blue-100 rounded-md px-1.5 py-0.5 transition-colors"
                        onClick={(e) => { e.stopPropagation(); setEnsembleMode(true); setEnsembleChecked([]); }}
                      >
                        <Layers size={9} /> 多选融合
                      </button>
                    ) : (
                      <button
                        className="text-[9px] font-bold text-slate-500 hover:text-slate-600 flex items-center gap-0.5 bg-slate-100 hover:bg-slate-200 rounded-md px-1.5 py-0.5 transition-colors"
                        onClick={(e) => { e.stopPropagation(); setEnsembleMode(false); setEnsembleChecked([]); }}
                      >
                        退出
                      </button>
                    )}
                  </div>
                  {ensembleMode && ensembleChecked.length >= 2 && (
                    <div className="px-2 pb-1.5">
                      <Button
                        type="primary" size="small" block
                        icon={<Layers size={10} />}
                        className="rounded-lg bg-blue-600 border-none font-bold text-[10px]"
                        onClick={() => setShowEnsembleModal(true)}
                      >
                        创建融合模型 ({ensembleChecked.length})
                      </Button>
                    </div>
                  )}
                  {displayModels.map(model => (
                    <ModelCard
                      key={model.model_id}
                      model={model}
                      isSelected={selectedId === model.model_id}
                      onClick={() => {
                        if (ensembleMode) {
                          const next = ensembleChecked.includes(model.model_id)
                            ? ensembleChecked.filter(id => id !== model.model_id)
                            : [...ensembleChecked, model.model_id];
                          setEnsembleChecked(next);
                        } else {
                          setSelectedId(model.model_id);
                        }
                      }}
                      onSetDefault={() => void handleSetDefaultById(model.model_id)}
                      canSetDefault={!model.is_default && model.status !== 'archived'}
                      showCheckbox={ensembleMode}
                      isChecked={ensembleMode && ensembleChecked.includes(model.model_id)}
                    />
                  ))}
                </>
              )}
            </div>
            {/* 底部操作 */}
            <div className="px-4 pb-4 pt-3 border-t border-slate-50">
              <Button
                type="primary" block
                icon={<Brain size={14} />}
                className="rounded-xl h-10 bg-slate-900 border-none font-black text-xs"
                onClick={() => navigate('/model-training')}
              >训练新模型</Button>
            </div>
          </div>
          {/* ═══ 右侧主区 ═══ */}
          <div className="flex-1 overflow-y-auto custom-scrollbar h-full bg-white relative">
            <AnimatePresence mode="wait">
              {!selectedModel ? (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full flex flex-col items-center justify-center gap-5 text-center"
                >
                  <div className="w-16 h-16 rounded-3xl bg-slate-100 flex items-center justify-center">
                    <Layers size={30} className="text-slate-300" />
                  </div>
                  <div>
                    <p className="font-black text-slate-300 tracking-widest text-sm">请选择模型</p>
                    <p className="text-xs text-slate-300 mt-1">从左侧选择模型查看详情</p>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key={selectedModel.model_id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="p-8 max-w-5xl mx-auto"
                >
                  {/* ── 模型 Header ── */}
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <div className="flex flex-wrap items-center gap-2 mb-2">
                        {/* 状态 badge */}
                        <span className={clsx(
                          'px-2.5 py-1 rounded-xl text-[9px] font-black uppercase tracking-wider flex items-center gap-1 border',
                          getStatusConfig(selectedModel.status).bg,
                          getStatusConfig(selectedModel.status).color,
                          getStatusConfig(selectedModel.status).border,
                        )}>
                          {getStatusConfig(selectedModel.status).icon}
                          {getStatusConfig(selectedModel.status).label}
                        </span>
                        {selectedModel.is_default && (
                          <span className="flex items-center gap-1 px-2.5 py-1 rounded-xl text-[9px] font-black bg-amber-50 text-amber-600 border border-amber-200">
                            <Star size={9} fill="currentColor" /> 默认
                          </span>
                        )}
                        <Tag className="rounded-xl border-none text-[9px] font-black uppercase bg-cyan-50 text-cyan-600 m-0">
                          {extractModelType(selectedModel)}
                        </Tag>
                        {getMeta(selectedModel).feature_count && (
                          <Tag className="rounded-xl border-none text-[9px] font-black bg-purple-50 text-purple-600 m-0">
                            {getMeta(selectedModel).feature_count} 维
                          </Tag>
                        )}
                      </div>
                      <h2 className="text-2xl font-black text-slate-900 tracking-tight m-0 font-mono leading-tight">
                        {modelDisplayName(selectedModel)}
                      </h2>
                      <p className="text-xs text-slate-400 mt-1 font-mono">
                        {selectedModel.model_id}
                        {selectedModel.created_at && ` · 创建于 ${dayjs(selectedModel.created_at).format('YYYY-MM-DD')}`}
                      </p>
                      {getMeta(selectedModel).description && (
                        <p className="text-xs text-slate-500 mt-2 max-w-xl">{getMeta(selectedModel).description}</p>
                      )}
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                      <Button
                        icon={<Code size={13} />}
                        className="rounded-xl h-9 px-4 font-bold border-slate-200 text-xs"
                        onClick={() => setShowConfigModal(true)}
                      >配置</Button>
                      {selectedModel.status !== 'archived' && (
                        <>
                          <Button
                            icon={<Archive size={13} />}
                            className="rounded-xl h-9 px-4 font-bold border-slate-200 text-xs text-slate-500"
                            onClick={handleArchive}
                            loading={archiving}
                          >归档</Button>
                          {!selectedModel.is_default && (
                            <Button
                              type="primary"
                              icon={<Star size={13} />}
                              className="rounded-xl h-9 px-5 font-black bg-slate-900 border-none text-xs"
                              onClick={handleSetDefault}
                              loading={settingDefault}
                            >设为默认</Button>
                          )}
                        </>
                      )}
                      {selectedModel.status === 'archived' && !isSystemModel(selectedModel) && (
                        <Button
                          danger
                          icon={<Trash2 size={13} />}
                          className="rounded-xl h-9 px-4 font-bold text-xs"
                          onClick={handleDeleteArchived}
                          loading={deleting}
                        >永久删除</Button>
                      )}
                    </div>
                  </div>
                  {/* ── 主 Tabs ── */}
                  <Tabs
                    activeKey={mainTab}
                    onChange={handleTabChange}
                    className="model-main-tabs"
                    items={[
                      {
                        key: 'detail',
                        label: <span className="text-xs font-black uppercase tracking-widest px-1">模型详情</span>,
                        children: <ModelDetailPanel model={selectedModel} />,
                      },
                      ...(selectedModel.source_run_id ? [{
                        key: 'training',
                        label: <span className="text-xs font-black uppercase tracking-widest px-1">训练溯源</span>,
                        children: (
                          <TrainingSourcePanel
                            model={selectedModel}
                            trainingRun={trainingRun}
                            loading={trainingRunLoading}
                          />
                        ),
                      }] : []),
                      {
                        key: 'attribution',
                        label: (
                          <span className="text-xs font-black uppercase tracking-widest px-1 flex items-center">
                            归因分析
                          </span>
                        ),
                        children: (
                          <AttributionAnalysisPanel
                            model={selectedModel}
                            shapSummary={shapSummary}
                            loading={shapLoading}
                            error={shapError}
                            featureLabelMap={featureLabelMap}
                            onRefresh={() => {
                              if (selectedModel) {
                                void loadFeatureLabelCatalog();
                                void loadShapSummary(selectedModel.model_id);
                              }
                            }}
                          />
                        ),
                      },
                      {
                        key: 'inference',
                        label: (
                          <span className="text-xs font-black uppercase tracking-widest px-1 flex items-center gap-1.5">
                            <Cpu size={11} />推理中心
                          </span>
                        ),
                        children: (
                          <div>
                            <div className="flex items-center gap-2 mb-4">
                              <Button
                                size="small"
                                type={inferenceMode === 'single' ? 'primary' : 'default'}
                                className={clsx('rounded-lg text-[10px] font-bold h-7 px-4', inferenceMode === 'single' ? 'bg-blue-600 border-blue-600' : 'border-slate-200')}
                                onClick={() => setInferenceMode('single')}
                              >
                                单日推理
                              </Button>
                              <Button
                                size="small"
                                type={inferenceMode === 'batch' ? 'primary' : 'default'}
                                className={clsx('rounded-lg text-[10px] font-bold h-7 px-4', inferenceMode === 'batch' ? 'bg-violet-600 border-violet-600' : 'border-slate-200')}
                                onClick={() => setInferenceMode('batch')}
                              >
                                批量多日
                              </Button>
                              <Button
                                size="small"
                                type={inferenceMode === 'batch-range' ? 'primary' : 'default'}
                                className={clsx('rounded-lg text-[10px] font-bold h-7 px-4', inferenceMode === 'batch-range' ? 'bg-emerald-600 border-emerald-600' : 'border-slate-200')}
                                onClick={() => setInferenceMode('batch-range')}
                              >
                                批量单日
                              </Button>
                              <Button
                                size="small"
                                type={inferenceMode === 'history' ? 'primary' : 'default'}
                                className={clsx('rounded-lg text-[10px] font-bold h-7 px-4', inferenceMode === 'history' ? 'bg-indigo-600 border-indigo-600' : 'border-slate-200')}
                                onClick={() => setInferenceMode('history')}
                              >
                                推理历史
                              </Button>
                            </div>
                            {inferenceMode === 'single' ? (
                              <InferenceCenterPanel
                                model={selectedModel}
                                inferenceDate={inferenceDate}
                                onDateChange={setInferenceDate}
                                targetDate={targetDate}
                                targetDateLoading={inferenceTargetLoading}
                                horizonDays={horizonDays}
                                running={inferenceRunning}
                                onRun={handleRunInference}
                                onRunAsDefault={handleSetDefault}
                                isDefault={selectedModel.is_default}
                                lastRun={lastInferenceRun}
                                history={inferenceHistory}
                                historyLoading={inferenceHistoryLoading}
                                autoSettings={autoSettings}
                                autoSaving={autoSaving}
                                onToggleAuto={handleToggleAuto}
                                latestInferenceRun={latestInferenceRun}
                                latestInferenceRunLoading={latestInferenceRunLoading}
                                precheck={inferencePrecheck}
                                precheckLoading={inferencePrecheckLoading}
                                onRefreshPrecheck={() => {
                                  if (selectedModel) {
                                    void loadPrecheck(selectedModel.model_id, inferenceDate?.format('YYYY-MM-DD'));
                                  }
                                }}
                                historyRunIdFilter={historyRunIdFilter}
                                onHistoryRunIdFilterChange={setHistoryRunIdFilter}
                                historyStatusFilter={historyStatusFilter}
                                onHistoryStatusFilterChange={setHistoryStatusFilter}
                                historyDateFilter={historyDateFilter}
                                onHistoryDateFilterChange={setHistoryDateFilter}
                                onDeleteHistory={handleDeleteHistory}
                              />
                            ) : inferenceMode === 'batch' ? (
                              <BatchInferencePanel modelId={selectedModel.model_id} horizonDays={horizonDays} />
                            ) : inferenceMode === 'batch-range' ? (
                              <BatchSingleDayPanel modelId={selectedModel.model_id} horizonDays={horizonDays} />
                            ) : (
                              <InferenceHistoryPanel
                                modelId={selectedModel.model_id}
                                onDelete={handleDeleteHistory}
                              />
                            )}
                          </div>
                        ),
                      },
                      {
                        key: 'backtest',
                        label: (
                          <span className="text-xs font-black uppercase tracking-widest px-1 flex items-center gap-1.5">
                            <BarChart3 size={11} />推理回测
                          </span>
                        ),
                        children: <InferenceBacktestModule modelId={selectedModel.model_id} />,
                      },
                      {
                        key: 'stock-picking',
                        label: (
                          <span className="text-xs font-black uppercase tracking-widest px-1 flex items-center gap-1.5">
                            <TrendingUp size={11} />选股
                          </span>
                        ),
                        children: <StockPickingPanel />,
                      },
                      {
                        key: 'negative-score',
                        label: (
                          <span className="text-xs font-black uppercase tracking-widest px-1 flex items-center gap-1.5">
                            <TrendingDown size={11} />负分参考
                          </span>
                        ),
                        children: <NegativeScorePanel />,
                      },
                      {
                        key: 'inference-research',
                        label: (
                          <span className="text-xs font-black uppercase tracking-widest px-1 flex items-center gap-1.5">
                            <History size={11} />推理研究
                          </span>
                        ),
                        children: (
                          <InferenceHistoryPanel
                            modelId={selectedModel.model_id}
                            onDelete={handleDeleteHistory}
                          />
                        ),
                      },
                      {
                        key: 'score-research',
                        label: (
                          <span className="text-xs font-black uppercase tracking-widest px-1 flex items-center gap-1.5">
                            <BarChart3 size={11} />模型分数研究
                          </span>
                        ),
                        children: <ModelScoreResearch modelId={selectedModel.model_id} />,
                      },
                    ]}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
      {/* ═══ 配置 Modal ═══ */}
      <Modal
        title={null}
        open={showConfigModal}
        onCancel={() => setShowConfigModal(false)}
        footer={null}
        width={820}
        centered
        styles={{ 
          body: { padding: 0 },
          mask: { backdropFilter: 'blur(4px)', backgroundColor: 'rgba(0,0,0,0.2)' }
        }}
        className="config-modal-container"
      >
        <div className="bg-white rounded-2xl overflow-hidden flex flex-col">
          {/* Header */}
          <div className="px-8 py-6 border-b border-slate-50 flex items-center justify-between bg-white">
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 bg-blue-50 rounded-2xl flex items-center justify-center text-blue-600 shadow-sm border border-blue-100">
                <Code size={20} />
              </div>
              <div>
                <h3 className="text-lg font-black text-slate-800 m-0 tracking-tight flex items-center gap-2">
                  配置文件浏览器
                  <span className="px-2 py-0.5 bg-slate-100 text-[9px] font-black text-slate-400 rounded-md tracking-widest">只读</span>
                </h3>
                <p className="text-[10px] text-slate-400 font-mono mt-0.5">{selectedModel?.model_id}</p>
              </div>
            </div>
            <button 
              onClick={() => setShowConfigModal(false)}
              className="p-2 hover:bg-slate-50 rounded-xl text-slate-300 hover:text-slate-500 transition-all"
            >
              <XCircle size={20} />
            </button>
          </div>
          {/* Tab Selector */}
          <div className="px-8 pt-4">
            <div className="flex bg-slate-50 p-1 rounded-xl w-fit">
              <button
                onClick={() => setActiveConfigTab('meta')}
                className={clsx(
                  "px-4 py-1.5 text-[10px] font-black tracking-widest rounded-lg transition-all",
                  activeConfigTab === 'meta' ? "bg-white text-blue-600 shadow-sm" : "text-slate-400 hover:text-slate-600"
                )}
              >
                元数据
              </button>
              <button
                onClick={() => setActiveConfigTab('metrics')}
                className={clsx(
                  "px-4 py-1.5 text-[10px] font-black tracking-widest rounded-lg transition-all",
                  activeConfigTab === 'metrics' ? "bg-white text-blue-600 shadow-sm" : "text-slate-400 hover:text-slate-600"
                )}
              >
                指标
              </button>
            </div>
          </div>
          {/* Code Area */}
          <div className="p-8">
            <div className="bg-slate-900 rounded-2xl p-6 border border-slate-800 shadow-inner relative group">
              <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                 <Tag className="bg-slate-800 border-slate-700 text-slate-400 text-[8px] font-mono">格式</Tag>
              </div>
              <pre className="text-[11px] font-mono text-emerald-400 leading-relaxed whitespace-pre-wrap max-h-[420px] overflow-auto custom-scrollbar scrollbar-dark">
                {activeConfigTab === 'meta'
                  ? JSON.stringify(selectedModel?.metadata_json ?? {}, null, 2)
                  : JSON.stringify(selectedModel?.metrics_json ?? {}, null, 2)}
              </pre>
            </div>
            {/* Actions */}
            <div className="mt-6 flex justify-between items-center">
              <div className="flex items-center gap-2 text-[10px] text-slate-400 font-bold">
                <Shield size={12} className="text-blue-500" />
                资产受保护资源，仅供审计查看
              </div>
              <div className="flex gap-3">
                <Button 
                  className="rounded-xl h-10 px-6 font-bold border-slate-100 text-slate-500 hover:bg-slate-50" 
                  onClick={() => setShowConfigModal(false)}
                >
                  关闭
                </Button>
                <Button 
                  type="primary" 
                  icon={<Download size={14} />}
                  className="rounded-xl h-10 px-8 font-black bg-blue-600 border-none shadow-lg shadow-blue-200" 
                  onClick={() => {
                    const txt = activeConfigTab === 'meta'
                      ? JSON.stringify(selectedModel?.metadata_json ?? {}, null, 2)
                      : JSON.stringify(selectedModel?.metrics_json ?? {}, null, 2);
                    navigator.clipboard.writeText(txt);
                    message.success('已复制到剪贴板');
                  }}
                >
                  复制内容
                </Button>
              </div>
            </div>
          </div>
        </div>
      </Modal>

      {/* 多模型融合创建对话框 */}
      <CreateEnsembleModal
        open={showEnsembleModal}
        onCancel={() => setShowEnsembleModal(false)}
        onCreated={(newModelId) => {
          setEnsembleMode(false);
          setEnsembleChecked([]);
          setSelectedId(newModelId);
          void loadModels(true);
        }}
        models={userModels.filter(m => ensembleChecked.includes(m.model_id))}
      />
    </div>
  );
};

export default ModelRegistryPage;
