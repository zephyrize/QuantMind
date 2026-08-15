import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Spin, message, Result, Button, Space, Typography, Tag, Progress, List, Badge, Divider } from 'antd';
import { 
    UserOutlined, 
    LineChartOutlined, 
    MessageOutlined, 
    HeartOutlined, 
    LoginOutlined, 
    HomeOutlined,
    ThunderboltOutlined,
    DeploymentUnitOutlined,
    DatabaseOutlined,
    GlobalOutlined,
    ApiOutlined,
    SwapOutlined,
    CheckCircleFilled,
    ClockCircleOutlined,
    AreaChartOutlined
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { adminService } from '../services/adminService';
import { authService } from '../../auth/services/authService';
import { useAppDispatch } from '../../../store';
import { logout } from '../../auth/store/authSlice';
import { DashboardMetrics, DashboardServiceInfo } from '../types';

const { Title, Text } = Typography;

export const AdminDashboard: React.FC = () => {
    const dispatch = useAppDispatch();
    const navigate = useNavigate();
    const location = useLocation();
    const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
    const [loading, setLoading] = useState(true);
    const [authError, setAuthError] = useState<{ status: number; message: string } | null>(null);

    useEffect(() => {
        loadMetrics();
    }, []);

    const loadMetrics = async () => {
        try {
            adminService.clearMetricsUnauthorized();
            setAuthError(null);
            const data = await adminService.getMetrics();
            setMetrics(data);
        } catch (err: any) {
            const status = err?.response?.status;
            const isLocked = String(err?.message || '').includes('ADMIN_METRICS_UNAUTHORIZED_LOCKED');
            const isAuthError = isLocked || status === 401 || status === 403 || (axios.isAxiosError(err) && (err.response?.status === 401 || err.response?.status === 403));
            
            if (isAuthError) {
                adminService.markMetricsUnauthorized();
                setAuthError({
                    status: status || 401,
                    message: status === 403 ? '您没有访问管理面板的权限。' : '您的登录会话已过期，请重新登录。'
                });
                return;
            }
            message.error('加载系统指标失败');
        } finally {
            setLoading(false);
        }
    };

    if (authError) {
        return (
            <div className="flex items-center justify-center py-20 bg-white border border-slate-200 rounded-3xl shadow-sm">
                <Result
                    status="403"
                    title={<span className="text-xl font-bold text-slate-800">访问受限</span>}
                    subTitle={<span className="text-slate-500">{authError.message}</span>}
                    extra={[
                        <Button 
                            type="primary" 
                            key="login" 
                            icon={<LoginOutlined />}
                            size="large"
                            className="h-11 rounded-xl px-8 bg-slate-900 border-none shadow-sm"
                            onClick={async () => {
                                await dispatch(logout());
                                navigate('/auth/login', { state: { from: location } });
                            }}
                        >
                            重新登录
                        </Button>,
                        <Button 
                            key="home" 
                            icon={<HomeOutlined />}
                            size="large"
                            className="h-11 rounded-xl px-8 text-slate-600 font-bold hover:bg-slate-50 transition-all border-slate-200"
                            onClick={() => navigate('/')}
                        >
                            返回首页
                        </Button>
                    ]}
                />
            </div>
        );
    }

    if (loading || !metrics) return (
        <div className="w-full flex flex-col items-center justify-center py-32 space-y-4">
            <Spin size="large" />
            <Text className="text-slate-400 font-bold text-xs">正在加载指标数据...</Text>
        </div>
    );

    const serviceStats: DashboardServiceInfo[] = metrics.system?.services || [];

    const iconMap: Record<string, React.ReactNode> = {
        api: <ApiOutlined />,
        engine: <ThunderboltOutlined />,
        trade: <SwapOutlined />,
        stream: <GlobalOutlined />,
        postgres: <DatabaseOutlined />,
        redis: <DatabaseOutlined />,
        data_gateway: <DeploymentUnitOutlined />,
        web: <HomeOutlined />,
        qwenpaw: <MessageOutlined />,
        rsshub: <GlobalOutlined />,
        huntly: <MessageOutlined />,
        dashboard: <AreaChartOutlined />,
        celery: <ThunderboltOutlined />,
        celery_beat: <ClockCircleOutlined />,
    };

    const serviceDescMap: Record<string, string> = {
        api: '用户认证 · 策略管理 · 社区',
        engine: 'Qlib回测 · AI策略 · 模型推理',
        trade: '订单管理 · 持仓 · 风控',
        stream: '实时行情 · WebSocket推送',
    };

    const servicePortMap: Record<string, string> = {
        api: '8000',
        engine: '8001',
        trade: '8002',
        stream: '8003',
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <div className="flex items-center justify-between mb-2">
                <div>
                    <Title level={4} className="!m-0 !font-black !text-slate-800 text-lg">系统控制台</Title>
                    <Text className="text-slate-400 text-xs font-medium">基础设施节点监控与管理</Text>
                </div>
                <Button
                    icon={<ThunderboltOutlined />}
                    onClick={loadMetrics}
                    className="rounded-xl font-bold bg-white text-slate-800 border-slate-200 hover:border-slate-800 hover:text-slate-800 shadow-sm h-10 px-6"
                >
                    刷新数据
                </Button>
            </div>

            {/* Core Services Grid */}
            <Row gutter={[20, 20]}>
                {serviceStats.map((s, idx) => {
                    const isHealthy = s.healthy && s.status === 'healthy';
                    const isUnreachable = s.status === 'unreachable';
                    const isDisabled = s.status === 'disabled';
                    return (
                        <Col xs={24} sm={12} lg={8} xl={6} key={s.service || idx}>
                            <Card className="rounded-2xl border-slate-200 shadow-sm hover:shadow-md transition-all">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-10 h-10 rounded-xl ${isHealthy ? 'bg-emerald-50' : isUnreachable ? 'bg-rose-50' : isDisabled ? 'bg-slate-50' : 'bg-amber-50'} flex items-center justify-center ${isHealthy ? 'text-emerald-600' : isUnreachable ? 'text-rose-500' : isDisabled ? 'text-slate-400' : 'text-amber-500'} border ${isHealthy ? 'border-emerald-100' : isUnreachable ? 'border-rose-100' : isDisabled ? 'border-slate-100' : 'border-amber-100'}`}>
                                            {iconMap[s.service] || <ApiOutlined />}
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-1.5">
                                                <Text className="font-black text-slate-800 text-sm">{s.service.toUpperCase()}</Text>
                                                <Badge status={isHealthy ? 'processing' : isDisabled ? 'default' : 'error'} color={isHealthy ? '#10b981' : isDisabled ? '#94a3b8' : '#ef4444'} />
                                            </div>
                                            <Text className="text-[10px] text-slate-400 font-bold">
                                                {s.port ? `端口 ${s.port}` : s.service === 'celery' ? '异步任务' : s.service === 'celery_beat' ? '定时调度' : `端口 ${servicePortMap[s.service] || '—'}`}
                                            </Text>
                                        </div>
                                    </div>
                                    <Tag color={isHealthy ? 'success' : isUnreachable ? 'error' : isDisabled ? 'default' : 'warning'} className="m-0 border-none rounded-full px-2 text-[9px] font-black">
                                        {isHealthy ? '运行中' : isUnreachable ? '不可达' : isDisabled ? '未启用' : '异常'}
                                    </Tag>
                                </div>
                                <div className="space-y-1.5">
                                    <div className="flex justify-between items-center text-[10px] font-black mb-1">
                                        <span className="text-slate-400">健康评分</span>
                                        <span className={isDisabled ? "text-slate-400" : s.score < 60 ? "text-rose-500" : s.score < 90 ? "text-amber-500" : "text-emerald-600"}>{isDisabled ? '—' : `${s.score}%`}</span>
                                    </div>
                                    <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-1000 ${isDisabled ? 'bg-slate-300' : s.score < 60 ? 'bg-rose-500' : s.score < 90 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                            style={{ width: `${s.score}%` }}
                                        />
                                    </div>
                                    <Text className="text-[10px] text-slate-400 font-medium block pt-1">{s.desc || serviceDescMap[s.service] || s.url || '—'}</Text>
                                </div>
                            </Card>
                        </Col>
                    );
                })}
            </Row>

            <Divider className="!m-0 border-slate-100" />

            <Row gutter={[24, 24]}>
                {/* Main Stats */}
                <Col span={24} lg={16}>
                    <div className="space-y-6">
                        <Title level={5} className="!m-0 !font-black !text-slate-800 text-xs opacity-50">全局统计</Title>
                        <Row gutter={[20, 20]}>
                            {[
                                { title: "总用户数", value: metrics.users.total, sub: `今日新增 ${metrics.users.new_today} 人`, icon: <UserOutlined /> },
                                { title: "模拟策略", value: metrics.strategies.live, sub: `共 ${metrics.strategies.total} 个策略`, icon: <LineChartOutlined /> },
                                { title: "数据记录", value: metrics.content.posts, sub: "社区互动数据", icon: <DatabaseOutlined /> },
                                { title: "系统运行", value: metrics.system.uptime_days, suffix: "天", sub: `健康度: ${metrics.system.health_score}%`, icon: <HeartOutlined /> }
                            ].map((item, idx) => (
                                <Col span={12} key={idx}>
                                    <Card className="rounded-2xl border-slate-100 bg-white shadow-sm">
                                        <Statistic 
                                            title={<span className="text-[10px] font-black text-slate-400">{item.title}</span>}
                                            value={item.value}
                                            suffix={item.suffix}
                                            valueStyle={{ fontWeight: 900, color: '#1e293b', fontSize: '24px', letterSpacing: '-0.025em' }}
                                            prefix={<div className="text-slate-300 mr-2">{item.icon}</div>}
                                        />
                                        <div className="mt-2 text-[11px] font-bold text-slate-400 flex items-center gap-1">
                                            <div className="w-1 h-1 rounded-full bg-slate-200" />
                                            {item.sub}
                                        </div>
                                    </Card>
                                </Col>
                            ))}
                        </Row>
                        
                        <Card className="rounded-2xl border-slate-100 shadow-sm" title={<span className="text-xs font-black text-slate-500">节点性能历史</span>}>
                            <div className="py-12 flex flex-col items-center justify-center bg-slate-50 rounded-xl border border-dashed border-slate-200">
                                <AreaChartOutlined className="text-slate-300 text-3xl mb-3" />
                                <Text className="text-slate-400 font-bold text-xs">实时吞吐量数据收集中...</Text>
                            </div>
                        </Card>
                    </div>
                </Col>

                {/* Side Activity */}
                <Col span={24} lg={8}>
                    <div className="space-y-6">
                        <Title level={5} className="!m-0 !font-black !text-slate-800 text-xs opacity-50">最近事件</Title>
                        <Card className="rounded-2xl border-slate-200 shadow-sm p-2">
                            {metrics.recent_events && metrics.recent_events.length > 0 ? (
                                <>
                                    <List
                                        itemLayout="horizontal"
                                        dataSource={metrics.recent_events}
                                        renderItem={(item: any) => (
                                            <List.Item className="!px-4 !py-3 hover:bg-slate-50 rounded-xl transition-all cursor-pointer">
                                                <List.Item.Meta
                                                    avatar={
                                                        <div className={`mt-1.5 w-2 h-2 rounded-full ${
                                                            item.type === 'success' ? 'bg-emerald-500' : 
                                                            item.type === 'warning' ? 'bg-rose-500' : 'bg-blue-500'
                                                        }`} />
                                                    }
                                                    title={<span className="text-xs font-bold text-slate-700">{item.title}</span>}
                                                    description={<span className="text-[10px] text-slate-400 font-bold">{item.time}</span>}
                                                />
                                            </List.Item>
                                        )}
                                    />
                                    <div className="p-4 pt-2">
                                        <Button block className="rounded-xl border-slate-200 text-slate-500 font-bold text-xs h-10 hover:border-slate-800 hover:text-slate-800">
                                            查看审计日志
                                        </Button>
                                    </div>
                                </>
                            ) : (
                                <div className="py-12 flex flex-col items-center justify-center">
                                    <ClockCircleOutlined className="text-slate-300 text-3xl mb-3" />
                                    <Text className="text-slate-400 font-bold text-xs">暂无事件记录</Text>
                                </div>
                            )}
                        </Card>
                    </div>
                </Col>
            </Row>
        </div>
    );
};
