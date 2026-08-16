import React, { useEffect, useMemo, useReducer, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Brain, ChevronRight, Play, Settings2, BarChart, Database,
  Copy, Sparkles, RefreshCcw, Target
} from 'lucide-react';
import {
  Button, Space, Tag, Typography, message, Card
} from 'antd';
import dayjs, { Dayjs } from 'dayjs';
import { clsx } from 'clsx';
import { PAGE_LAYOUT } from '../config/pageLayout';
import { modelTrainingService } from '../services/modelTrainingService';
import { useAppSelector } from '../store';
import { selectCurrentMarket, AppMarket } from '../store/slices/uiSlice';
import { getMarketConfig } from '../config/marketConfig';
import { TrainingTarget, TrainingParams, TrainingContext, TrainingStatus, TrainingDraft, SplitKey, TimePeriodMap, FeatureCategory, STORAGE_KEY, DEFAULT_FEATURE_CATEGORIES, PRESET_DEFAULT_FEATURES, MARKET_DEFAULT_FEATURES, getDefaultFeaturesForMarket, resolveDefaultSelectedFeatures, DEFAULT_TIME_PERIODS, DEFAULT_TARGET, DEFAULT_PARAMS, DEFAULT_CONTEXT, buildAutoDisplayName, buildLabelFormula, buildEffectiveTradeDate, daysBetween, toISOStringRange, restoreRange, shouldMigrateLegacyDraftPeriods, buildTrainingRequest, formatRange, toDynamicCategories, TrainingResult, buildBackendTrainingPayload, parseTrainingResult, parseSuggestedTimePeriods, MODEL_DL_DEFAULTS, WfaConfig, FeatureMode } from './training/trainingUtils';
import { AdminModelFeatureDataCoverage } from '../features/admin/types';
import { adminService } from '../features/admin/services/adminService';
import { FeatureSelector } from './training/FeatureSelector';
import { TrainingTargetConfig } from './training/TrainingTargetConfig';
import { ParameterConfig } from './training/ParameterConfig';
import { TrainingConsole } from './training/TrainingConsole';
import { TrainingResultView } from './training/TrainingResultView';

const { Title, Paragraph } = Typography;

const TRAINING_MODULES = [
  { title: '特征选择', description: '筛选输入因子', icon: Database, hint: '第一步' },
  { title: '训练目标', description: '定义 T+N 标签口径', icon: Target, hint: '第二步' },
  { title: '参数配置', description: '设置超参与训练上下文', icon: Settings2, hint: '第三步' },
  { title: '执行训练', description: '编排请求与日志预览', icon: Play, hint: '第四步' },
  { title: '结果入库', description: '查看元数据与产物', icon: BarChart, hint: '第五步' },
];

const TRAINING_PAGE_BOTTOM_SAFE_CLASS = 'pb-[30px]';
let draftRestoreNoticeShown = false;

const MetricCard: React.FC<{
  label: string;
  value: string;
  hint?: string;
  centered?: boolean;
}> = ({ label, value, hint, centered = false }) => (
  <div className={clsx('rounded-2xl border border-slate-200 bg-white p-4 shadow-sm', centered && 'text-center')}>
    <div className={clsx('text-[10px] font-black uppercase tracking-[0.18em] text-slate-400', centered && 'text-center')}>{label}</div>
    <div className={clsx('mt-2 text-lg font-semibold text-slate-900', centered && 'text-center')}>{value}</div>
    {hint ? <div className={clsx('mt-1 text-xs text-slate-500', centered && 'text-center')}>{hint}</div> : null}
  </div>
);

// ==========================================================================
// P0-4: useReducer 草稿恢复（原子化 7 字段一次性写入）
// ==========================================================================

interface FormState {
  selectedFeatures: string[];
  featureMode: FeatureMode;
  timePeriods: TimePeriodMap;
  wfaConfig: WfaConfig;
  target: TrainingTarget;
  params: TrainingParams;
  context: TrainingContext;
  displayName: string;
  displayNameMode: 'auto' | 'manual';
  draftHydrated: boolean;
}

type FormAction =
  | { type: 'HYDRATE'; payload: TrainingDraft | null }
  | { type: 'SET_FEATURES'; payload: string[] }
  | { type: 'SET_FEATURE_MODE'; payload: FeatureMode }
  | { type: 'SET_TIME'; key: SplitKey; value: [Dayjs, Dayjs] }
  | { type: 'SET_TARGET'; payload: TrainingTarget }
  | { type: 'SET_PARAMS'; payload: TrainingParams }
  | { type: 'SET_CONTEXT'; payload: TrainingContext }
  | { type: 'SET_DISPLAY_NAME'; payload: { name: string; mode: 'auto' | 'manual' } }
  | { type: 'SET_WFA'; payload: WfaConfig }
  | { type: 'SET_FEATURE_CATEGORIES'; payload: FeatureCategory[] }
  | { type: 'SET_MARKET_CONTEXT'; payload: { market: AppMarket; benchmark: string } };

function formReducer(state: FormState, action: FormAction): FormState {
  switch (action.type) {
    case 'HYDRATE': {
      if (!action.payload) return { ...state, draftHydrated: true };
      const p = action.payload;
      const restoredParams = { ...DEFAULT_PARAMS, ...p.params };
      if (!p.params?.model_types && p.params?.model_type) {
        restoredParams.model_types = [p.params.model_type];
      }
      if (restoredParams.model_type && MODEL_DL_DEFAULTS[restoredParams.model_type]) {
        const defaults = MODEL_DL_DEFAULTS[restoredParams.model_type];
        Object.assign(restoredParams, defaults);
      }
      const restoredWfa = p.wfa ?? { enabled: false, strategy: 'rolling', nWindows: 4, trainYears: 3, valMonths: 12, stepMonths: 12 };
      return {
        ...state,
        featureMode: p.featureMode || "snapshot",
        selectedFeatures: p.selectedFeatures && p.selectedFeatures.length > 0
          ? p.selectedFeatures : state.selectedFeatures,
        timePeriods: {
          train: restoreRange(p.timePeriods?.train, DEFAULT_TIME_PERIODS.train),
          val: restoreRange(p.timePeriods?.val, DEFAULT_TIME_PERIODS.val),
          test: restoreRange(p.timePeriods?.test, DEFAULT_TIME_PERIODS.test),
        },
        target: p.target || DEFAULT_TARGET,
        params: restoredParams,
        context: { ...DEFAULT_CONTEXT, ...p.context },
        displayNameMode: p.displayNameMode || 'auto',
        displayName: p.displayName || state.displayName,
        wfaConfig: restoredWfa,
        draftHydrated: true,
      };
    }
    case 'SET_FEATURES':
      return { ...state, selectedFeatures: action.payload };
    case 'SET_FEATURE_MODE':
      return { ...state, featureMode: action.payload };
    case 'SET_TIME':
      return { ...state, timePeriods: { ...state.timePeriods, [action.key]: action.value } };
    case 'SET_TARGET':
      return { ...state, target: action.payload };
    case 'SET_PARAMS':
      return { ...state, params: action.payload };
    case 'SET_CONTEXT':
      return { ...state, context: action.payload };
    case 'SET_DISPLAY_NAME':
      return { ...state, displayName: action.payload.name, displayNameMode: action.payload.mode };
    case 'SET_WFA':
      return { ...state, wfaConfig: action.payload };
    case 'SET_FEATURE_CATEGORIES':
      return { ...state };
    case 'SET_MARKET_CONTEXT':
      return { ...state, context: { ...state.context, ...action.payload } };
    default:
      return state;
  }
}

// ==========================================================================
// Component
// ==========================================================================

export const ModelTrainingPage: React.FC = () => {
  const navigate = useNavigate();
  const currentMarket = useAppSelector(selectCurrentMarket);

  // ── useReducer：草稿持久化的 7 字段 ──
  const [formState, dispatch] = useReducer(formReducer, {
    selectedFeatures: getDefaultFeaturesForMarket(currentMarket),
    featureMode: "snapshot" as FeatureMode,
    timePeriods: DEFAULT_TIME_PERIODS,
    wfaConfig: { enabled: false, strategy: 'rolling', nWindows: 4, trainYears: 3, valMonths: 12, stepMonths: 12 },
    target: DEFAULT_TARGET,
    params: DEFAULT_PARAMS,
    context: DEFAULT_CONTEXT,
    displayName: buildAutoDisplayName(dayjs(), DEFAULT_TARGET, PRESET_DEFAULT_FEATURES.length),
    displayNameMode: 'auto' as const,
    draftHydrated: false,
  });

  // ── useState: 训练运行时 state（不参与草稿持久化） ──
  const [currentStep, setCurrentStep] = useState(0);
  const [featureCategories, setFeatureCategories] = useState<FeatureCategory[]>(DEFAULT_FEATURE_CATEGORIES);
  const [featureCatalogLoading, setFeatureCatalogLoading] = useState(false);
  const [dataCoverage, setDataCoverage] = useState<AdminModelFeatureDataCoverage | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>('draft');
  const [executionStage, setExecutionStage] = useState('待配置');
  const [backendRunStatus, setBackendRunStatus] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [resultError, setResultError] = useState<string>('');
  const [settingDefaultModel, setSettingDefaultModel] = useState(false);
  const [draftSavedAt, setDraftSavedAt] = useState<string>('');
  const [trainingNodes, setTrainingNodes] = useState<any[]>([]);
  const [selectedNode, setSelectedNode] = useState<string>('local');
  const [nodesLoading, setNodesLoading] = useState(false);

  const timersRef = useRef<number[]>([]);
  const pollTimerRef = useRef<number | null>(null);
  const logsRef = useRef<string[]>([]);
  const catalogSuggestionAppliedRef = useRef(false);

  // Derive individual fields from formState for inline use
  const { selectedFeatures, featureMode, timePeriods, wfaConfig, target, params, context, displayName, displayNameMode } = formState;

  const labelFormula = useMemo(() => buildLabelFormula(target), [target]);
  const effectiveTradeDate = useMemo(() => buildEffectiveTradeDate(target, timePeriods.test[0]), [target, timePeriods.test]);

  // 市场切换
  useEffect(() => {
    const mc = getMarketConfig(currentMarket);
    dispatch({ type: 'SET_MARKET_CONTEXT', payload: { market: currentMarket, benchmark: mc.benchmark } });
    dispatch({ type: 'SET_FEATURES', payload: getDefaultFeaturesForMarket(currentMarket) });
    catalogSuggestionAppliedRef.current = false;
  }, [currentMarket]);

  const featureCount = featureMode === "qlib_alpha158" ? 158 : selectedFeatures.length;
  const autoDisplayName = useMemo(
    () => buildAutoDisplayName(dayjs(), target, featureCount, undefined, currentMarket),
    [target, featureCount, currentMarket]
  );
  const trainDays = useMemo(() => daysBetween(timePeriods.train), [timePeriods.train]);
  const valDays = useMemo(() => daysBetween(timePeriods.val), [timePeriods.val]);
  const testDays = useMemo(() => daysBetween(timePeriods.test), [timePeriods.test]);
  const totalDays = trainDays + valDays + testDays;
  const requestPreview = useMemo(
    () => buildTrainingRequest(selectedFeatures, featureCategories, timePeriods, target, params, context, displayName, currentMarket, wfaConfig, featureMode),
    [selectedFeatures, featureMode, featureCategories, timePeriods, target, params, context, displayName, currentMarket, wfaConfig]
  );
  const isReadyToTrain = (featureMode === "qlib_alpha158" || selectedFeatures.length > 0) && target.horizonDays >= 1 && totalDays > 0;
  const isTrainingInProgress =
    trainingStatus === 'running' ||
    ['pending', 'provisioning', 'running', 'waiting_callback'].includes((backendRunStatus || '').toLowerCase());
  const disableStartTraining = isTrainingInProgress && currentStep === 3;

  // 自动 displayName
  useEffect(() => {
    if (displayNameMode !== 'auto') return;
    if (displayName !== autoDisplayName) {
      dispatch({ type: 'SET_DISPLAY_NAME', payload: { name: autoDisplayName, mode: 'auto' } });
    }
  }, [autoDisplayName, displayName, displayNameMode]);

  // 训练节点
  useEffect(() => {
    let active = true;
    const loadNodes = async () => {
      setNodesLoading(true);
      try {
        const resp = await adminService.listTrainingNodes();
        if (active && resp?.nodes) {
          setTrainingNodes(resp.nodes);
        }
      } catch { /* silent */ } finally {
        if (active) setNodesLoading(false);
      }
    };
    loadNodes();
    return () => { active = false; };
  }, []);

  // 特征字典加载
  useEffect(() => {
    let active = true;
    const loadCatalog = async () => {
      setFeatureCatalogLoading(true);
      try {
        const catalog = await modelTrainingService.getFeatureCatalog(currentMarket, false);
        if (!active) return;
        const dynamicCats = toDynamicCategories(catalog);
        setFeatureCategories(dynamicCats);
        dispatch({ type: 'SET_FEATURES', payload: resolveDefaultSelectedFeatures(dynamicCats, currentMarket) });
      } catch (error) {
        if (active) message.warning('特征字典加载失败，已回退到内置字段');
      } finally {
        if (active) setFeatureCatalogLoading(false);
      }

      try {
        const catalogWithCoverage = await modelTrainingService.getFeatureCatalog(currentMarket, true);
        if (!active) return;
        if (catalogWithCoverage.data_coverage) {
          setDataCoverage(catalogWithCoverage.data_coverage);
        }
        if (catalogWithCoverage.data_coverage?.suggested_periods && !catalogSuggestionAppliedRef.current) {
          const suggested = parseSuggestedTimePeriods(catalogWithCoverage.data_coverage.suggested_periods);
          if (suggested) {
            dispatch({ type: 'SET_TIME', key: 'train', value: suggested.train });
            dispatch({ type: 'SET_TIME', key: 'val', value: suggested.val });
            dispatch({ type: 'SET_TIME', key: 'test', value: suggested.test });
            catalogSuggestionAppliedRef.current = true;
          }
        }
      } catch { /* coverage failure ok */ }
    };
    loadCatalog();
    return () => { active = false; };
  }, [currentMarket]);

  // P0-4: 草稿恢复 — 一次 dispatch 原子化写入（替代 7 个 setState）
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) { dispatch({ type: 'HYDRATE', payload: null }); return; }
    try {
      const parsed = JSON.parse(saved) as TrainingDraft;
      dispatch({ type: 'HYDRATE', payload: parsed });
      if (!draftRestoreNoticeShown) {
        draftRestoreNoticeShown = true;
        message.success('已恢复上次训练草稿');
      }
    } catch {
      localStorage.removeItem(STORAGE_KEY);
      dispatch({ type: 'HYDRATE', payload: null });
    }
  }, []); // 只 mount 时执行

  // P0-4: 草稿保存 — draftHydrated 守卫防止覆盖已恢复草稿
  useEffect(() => {
    if (!formState.draftHydrated) return;
    const draft: TrainingDraft = {
      displayName,
      displayNameMode,
      selectedFeatures,
      featureMode,
      timePeriods: {
        train: toISOStringRange(timePeriods.train),
        val: toISOStringRange(timePeriods.val),
        test: toISOStringRange(timePeriods.test),
      },
      target,
      params,
      context,
      wfa: wfaConfig,
      lastSavedAt: new Date().toISOString(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
    setDraftSavedAt(draft.lastSavedAt);
  }, [formState.draftHydrated, displayName, displayNameMode, selectedFeatures, featureMode, timePeriods, target, params, context, wfaConfig]);

  const clearTimers = () => {
    timersRef.current.forEach(t => window.clearTimeout(t));
    timersRef.current = [];
    if (pollTimerRef.current) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      clearTimers();
    };
  }, []);

  const pushLog = (line: string) => {
    const next = [...logsRef.current, `[${dayjs().format('HH:mm:ss')}] ${line}`];
    logsRef.current = next;
    setLogs(next);
  };

  const startTraining = async () => {
    if (isTrainingInProgress) {
      message.warning('训练任务进行中，请稍候');
      return;
    }
    if (!isReadyToTrain) { message.warning('配置不完整'); return; }
    clearTimers();
    setResultError('');
    setResult(null);
    setTrainingStatus('running');
    setExecutionStage('准备训练请求');
    setProgress(5);
    pushLog(`正在提交训练请求：${displayName}`);

    try {
      const payload = buildBackendTrainingPayload(requestPreview, timePeriods, { nodeId: selectedNode });
      const { runId } = await modelTrainingService.runTraining(payload);
      pushLog(`提交成功，Run ID: ${runId}`);

      pollTimerRef.current = window.setInterval(async () => {
        const run = await modelTrainingService.getTrainingRun(runId);
        setBackendRunStatus(run.status || '');
        if (run.logs) {
           run.logs.split('\n').filter(Boolean).forEach(line => {
             if (!logsRef.current.some(l => l.includes(line))) pushLog(line);
           });
        }
        if (run.status === 'running') setProgress(Math.max(run.progress || 20, 20));

        if (run.isCompleted) {
          clearTimers();
          if (run.status === 'failed') {
            const errorMsg = (run.result as any)?.error || '训练失败';
            setResultError(errorMsg);
            setTrainingStatus('draft');
          } else {
            const parsed = parseTrainingResult(requestPreview, runId, run.result);
            if (parsed) {
              setResult(parsed);
              setResultError('');
              setTrainingStatus('completed');
              setProgress(100);
              setCurrentStep(4);
              message.success('训练完成');
            } else {
              setResultError('结果解析失败');
              setTrainingStatus('draft');
            }
          }
        }
      }, 3000);
    } catch (err: any) {
      message.error(`提交失败: ${err.message}`);
      setTrainingStatus('draft');
    }
  };

  const stepAction = () => {
    if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
      return;
    }
    if (currentStep === 3) {
      startTraining();
      return;
    }
    setCurrentStep(0);
    setTrainingStatus('draft');
    setResult(null);
    setResultError('');
  };

  const handleResetAll = () => {
    clearTimers();
    const features = featureCategories.length > 0
      ? resolveDefaultSelectedFeatures(featureCategories, currentMarket)
      : getDefaultFeaturesForMarket(currentMarket);
    dispatch({ type: 'SET_FEATURES', payload: features });
    dispatch({ type: 'SET_TIME',  key: 'train', value: DEFAULT_TIME_PERIODS.train });
    dispatch({ type: 'SET_TIME',  key: 'val',   value: DEFAULT_TIME_PERIODS.val });
    dispatch({ type: 'SET_TIME',  key: 'test',  value: DEFAULT_TIME_PERIODS.test });
    dispatch({ type: 'SET_TARGET', payload: DEFAULT_TARGET });
    dispatch({ type: 'SET_PARAMS', payload: DEFAULT_PARAMS });
    dispatch({ type: 'SET_CONTEXT', payload: DEFAULT_CONTEXT });
    dispatch({ type: 'SET_DISPLAY_NAME', payload: { name: '', mode: 'auto' } });
    dispatch({ type: 'SET_WFA', payload: { enabled: false, strategy: 'rolling', nWindows: 4, trainYears: 3, valMonths: 12, stepMonths: 12 } });
    setTrainingStatus('draft');
    setResult(null);
    setCurrentStep(0);
    localStorage.removeItem(STORAGE_KEY);
    message.info('配置已重置');
  };

  const handleSetDefaultModel = async () => {
    const id = result?.modelRegistration?.modelId || result?.modelId;
    if (!id) return;
    try {
      setSettingDefaultModel(true);
      await modelTrainingService.setDefaultModel(id);
      message.success('成功重置默认模型');
    } catch (e: any) { message.error(e.message); }
    finally { setSettingDefaultModel(false); }
  };

  const stepActionLabel = currentStep < 3 ? '下一步' : currentStep === 3 ? '开始训练' : '重新配置';
  const currentModule = TRAINING_MODULES[currentStep] || TRAINING_MODULES[0];
  const CurrentIcon = currentModule.icon;

  return (
    <div className={PAGE_LAYOUT.outerClass}>
      <div className={PAGE_LAYOUT.frameClass}>
        <header className={PAGE_LAYOUT.headerClass} style={{ height: `${PAGE_LAYOUT.headerHeight}px` }}>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div className="flex items-center gap-2.5 ml-1">
              <h1 className="text-xl font-bold text-slate-800 tracking-tight">QuantMind</h1>
              <div className="h-4 w-[1px] bg-slate-200 self-center" />
              <span className="text-sm font-medium text-slate-500">模型训练中心</span>
            </div>
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden">
          <aside className="bg-white border-r border-gray-200 flex flex-col shadow-sm" style={{ width: `${PAGE_LAYOUT.sidebarWidth}px` }}>
            <div className="flex-1 py-4 overflow-y-auto custom-scrollbar">
              <div className="px-6 mb-2">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">训练步骤</p>
              </div>
              <div className="space-y-1">
                {TRAINING_MODULES.map((m, i) => (
                  <button key={m.title} onClick={() => setCurrentStep(i)} className={clsx('relative w-full px-6 text-left py-3 flex items-center gap-3', currentStep === i ? 'bg-blue-50' : 'hover:bg-gray-50')}>
                    {currentStep === i && <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500 rounded-r-full" />}
                    <div className={clsx('w-9 h-9 rounded-xl flex items-center justify-center', currentStep === i ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-400')}>
                      <m.icon size={16} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm font-medium text-gray-900">{m.title}</div>
                      <div className="text-[10px] text-gray-500 truncate">{m.description}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
            <div className="p-4 border-t border-gray-100 space-y-3">
               <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
                  <div className="text-[10px] uppercase font-bold text-slate-400 mb-1">当前配置摘要</div>
                  <div className="text-xs font-semibold text-slate-700">T+{target.horizonDays} · {target.mode === 'classification' ? '分类' : '回归'}</div>
                  <div className="text-[10px] text-slate-400 mt-1 truncate">{labelFormula}</div>
               </div>
               <div className="rounded-xl bg-slate-50 p-3 border border-slate-100">
                  <div className="text-[10px] uppercase font-bold text-slate-400 mb-2">训练节点</div>
                  <div className="flex gap-2">
                    {trainingNodes.length > 0
                      ? trainingNodes.map((n) => (
                          <button
                            key={n.id}
                            onClick={() => setSelectedNode(n.id)}
                            className={clsx(
                              'flex-1 rounded-lg px-2 py-1.5 text-xs font-semibold border transition-all',
                              selectedNode === n.id
                                ? n.type === 'remote'
                                  ? 'bg-orange-50 border-orange-300 text-orange-700'
                                  : 'bg-blue-50 border-blue-300 text-blue-700'
                                : 'bg-white border-gray-200 text-gray-500 hover:border-gray-300'
                            )}
                          >
                            {n.name}
                          </button>
                        ))
                      : <div className="text-[11px] text-slate-400 w-full text-center py-1">仅本地训练</div>}
                  </div>
                  {selectedNode !== 'local' && (
                    <div className="mt-2 text-[10px] text-orange-600 leading-relaxed">
                      将推送特征快照到 AutoDL，远程 GPU 训练完成后模型自动回传本机。
                    </div>
                  )}
               </div>
               <div className="flex gap-2">
                 <Button size="small" block className="rounded-lg" onClick={() => message.success('草稿已保存')}>保存草稿</Button>
                 <Button size="small" block className="rounded-lg" onClick={handleResetAll} disabled={isTrainingInProgress}>重置</Button>
               </div>
            </div>
          </aside>

          <main className="flex-1 flex flex-col bg-gray-50/50 min-w-0">
            <div className={PAGE_LAYOUT.breadcrumbClass}>
              <div className="flex items-center gap-2 text-sm">
                <span className="text-gray-500">训练中心</span>
                <span className="text-gray-400">/</span>
                <span className="text-gray-800 font-medium">{currentModule.title}</span>
              </div>
            </div>

            <div className={`flex-1 overflow-y-auto overflow-x-hidden p-6 ${TRAINING_PAGE_BOTTOM_SAFE_CLASS}`}>
              <div className="max-w-6xl mx-auto space-y-4">
                <Card className="rounded-2xl border-gray-200 shadow-sm" styles={{ body: { padding: 20 } }}>
                  <div className="flex items-start justify-between">
                    <div>
                        <div className="flex items-center gap-2">
                          <CurrentIcon size={18} className="text-blue-500" />
                          <Title level={4} className="!mb-0">{currentModule.title}</Title>
                        </div>
                        <Paragraph className="!mb-0 !mt-2 text-gray-500 text-xs">{currentModule.description}</Paragraph>
                    </div>
                    <Space>
                      <Button icon={<RefreshCcw size={14}/>} className="rounded-xl h-9" onClick={handleResetAll} disabled={isTrainingInProgress}>清空</Button>
                      <Button type="primary" icon={<ChevronRight size={14}/>} className="rounded-xl h-9 bg-blue-600" onClick={stepAction} disabled={disableStartTraining}>
                        {stepActionLabel}
                      </Button>
                    </Space>
                  </div>
                </Card>

                <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
                    <MetricCard label="市场" value={getMarketConfig(currentMarket).label} centered />
                    <MetricCard label="特征数" value={`${featureCount}`} centered />
                    <MetricCard label="预测周期" value={`T+${target.horizonDays}`} hint={target.mode} centered />
                    <MetricCard label="数据集天数" value={`${totalDays}`} hint={`${trainDays}/${valDays}/${testDays}`} centered />
                    <MetricCard label="状态" value={trainingStatus === 'draft' ? '待配置' : trainingStatus === 'running' ? '训练中' : '已完成'} centered />
                </div>

                <AnimatePresence mode="wait">
                  <motion.div key={currentStep} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.2 }}>
                    {currentStep === 0 && <div className="space-y-4"><div className="rounded-2xl border border-slate-200 bg-white p-4"><div className="text-sm font-semibold text-slate-800">特征模式</div><div className="mt-3 flex flex-wrap gap-2"><button onClick={() => dispatch({ type: 'SET_FEATURE_MODE', payload: 'snapshot' })} className={clsx("rounded-lg border px-3 py-2 text-sm", featureMode === "snapshot" ? "border-blue-500 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-600")}>自定义快照特征</button><button onClick={() => dispatch({ type: 'SET_FEATURE_MODE', payload: 'qlib_alpha158' })} className={clsx("rounded-lg border px-3 py-2 text-sm", featureMode === "qlib_alpha158" ? "border-blue-500 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-600")}>Qlib 原生 Alpha158</button></div><p className="mt-2 text-xs text-slate-500">Alpha158 仅使用 Qlib 二进制 OHLCV + factor 数据，并固定使用 Qlib LightGBM；不会读取自定义特征快照。</p></div>{featureMode === "snapshot" ? <FeatureSelector categories={featureCategories} selectedFeatures={selectedFeatures} onChange={(f) => dispatch({ type: 'SET_FEATURES', payload: f })} loading={featureCatalogLoading} /> : <div className="rounded-2xl border border-blue-100 bg-blue-50 p-5 text-sm text-slate-700"><div className="font-semibold text-blue-800">已选择 Qlib Alpha158（158 个原生因子）</div><p className="mt-2">训练、预测与回测使用项目的 Qlib 数据目录；此模式不会因为扩展特征缺失而失败。</p></div>}</div>}
                    {currentStep === 1 && <TrainingTargetConfig target={target} timePeriods={timePeriods} onTargetChange={(t) => dispatch({ type: 'SET_TARGET', payload: t })} onTimeChange={(k, v) => dispatch({ type: 'SET_TIME', key: k, value: v })} dataCoverage={dataCoverage} wfa={wfaConfig} onWfaChange={(w) => dispatch({ type: 'SET_WFA', payload: w })} />}
                    {currentStep === 2 && <ParameterConfig params={params} context={context} onParamsChange={(p) => dispatch({ type: 'SET_PARAMS', payload: p })} onContextChange={(c) => dispatch({ type: 'SET_CONTEXT', payload: c })} displayName={displayName} onDisplayNameChange={(n, m) => dispatch({ type: 'SET_DISPLAY_NAME', payload: { name: n, mode: m } })} autoDisplayName={autoDisplayName} market={currentMarket} />}
                    {currentStep === 3 && <TrainingConsole trainingStatus={trainingStatus} executionStage={executionStage} progress={progress} logs={logs} backendRunStatus={backendRunStatus} result={result} requestPreview={requestPreview} totalDays={totalDays} trainDays={trainDays} valDays={valDays} testDays={testDays} target={target} />}
                    {currentStep === 4 && <TrainingResultView result={result} resultError={resultError} settingDefaultModel={settingDefaultModel} onSetDefaultModel={handleSetDefaultModel} trainingStatus={trainingStatus} />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default ModelTrainingPage;
