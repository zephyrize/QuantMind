# QuantMind dev 分支：运行、配置与兼容性改造

> 本文是当前 `dev` 分支的变更说明和开发运行指南。完整的平台功能、数据下载和历史部署说明已归档至 [旧版 README](docs/README_旧版.md)。

## 本分支解决了什么

这轮改造聚焦于“同一份代码能稳定运行在 Docker、本地 Python、浏览器开发服务器和 Electron 中”。此前，配置散落在 Compose、根 `.env` 和源码默认值中；前后端切换运行方式时，常会遇到错误的数据库主机、前端请求指向浏览器自身 localhost、镜像代码被宿主机目录覆盖，或 Docker-in-Docker 无法找到真实宿主机路径等问题。

本分支将运行配置分层，统一环境加载逻辑，并补齐前后端组合测试方案和数据库兼容迁移。

## 变更总览

| 范围 | 已完成的处理 |
| --- | --- |
| 环境变量 | 拆分基础设施与应用配置；按运行位置自动推导数据库、Redis 地址。 |
| 后端镜像 | 将后端及运行所需模块构建进镜像，移除运行时源码挂载和启动时重复安装。 |
| 前端联通 | Docker Web、Vite 浏览器开发和 Electron 使用各自正确的 API/WS 路径。 |
| 测试组合 | 明确三种前后端组合的启动、验证和恢复方法。 |
| 子容器编排 | 自动从当前容器的 bind mount 推导 Docker daemon 可见的项目根路径，并兼容 Windows 路径。 |
| 数据库 | 新库初始化使用与 ORM 一致的枚举值；旧库启动时执行幂等兼容迁移。 |
| OSS 可用性 | 可选服务不再拉低健康评分；无 LLM Key 时保留非 AI 功能；本地策略存储可用。 |

## 快速开始

### 1. 准备两份配置文件

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

先在两个文件中替换所有 `CHANGE_ME_...` 密码和密钥。生产环境必须使用随机且独立的强密钥，且不要将 `.env`、`backend/.env` 提交到 Git。

### 2. 完整容器化运行

```bash
docker compose up -d --build --force-recreate \
  quantmind celery-worker celery-beat web

docker compose ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:3080/health
```

基础设施和第三方服务可按需启动：

```bash
docker compose up -d db redis
docker compose up -d rsshub huntly qwenpaw
```

默认入口：Web `http://localhost:3080`，API 文档 `http://localhost:8000/docs`。

## 环境变量重新设计

### 配置职责

| 文件 | 负责内容 | 不应放入的内容 |
| --- | --- | --- |
| 根目录 `.env` | Docker Compose 插值、PostgreSQL/Redis 凭据、宿主机端口、镜像构建参数。 | `DB_HOST`、`REDIS_HOST`、`DATABASE_URL` 和应用密钥。 |
| `backend/.env` | 应用密钥、管理员、LLM、服务监听端口、存储、任务和可选服务开关。 | 与 Docker 拓扑绑定的数据库/Redis 主机名。 |
| `config/runtime.env` | 由管理界面维护的运行时密钥；优先级最高。 | 长期基础设施配置。 |

后端通过 `backend.shared.env_loader.bootstrap_environment()` 集中加载配置，优先级为：

```text
进程环境变量 > config/runtime.env > backend/.env > 根目录 .env > 自动推导默认值
```

自动推导规则：

| 运行方式 | `DB_HOST` | `REDIS_HOST` |
| --- | --- | --- |
| 本地 Python（`QUANTMIND_RUNTIME=local`） | `127.0.0.1` | `127.0.0.1` |
| Docker Compose（`QUANTMIND_RUNTIME=docker`） | `db` | `redis` |

连接参数会由 `POSTGRES_*` / `REDIS_*` 派生，未显式配置 `DATABASE_URL` 时自动生成带 URL 编码凭据的 SQLAlchemy 异步连接串。显式设置 `DB_HOST` 或 `REDIS_HOST` 时始终优先，适合外部数据库、CI 和测试环境。

### 常用配置项

| 配置项 | 位置 | 用途 |
| --- | --- | --- |
| `POSTGRES_DB`、`POSTGRES_USER`、`POSTGRES_PASSWORD` | 根 `.env` | PostgreSQL 容器与后端共用的凭据。 |
| `REDIS_PASSWORD` | 根 `.env` | Redis 容器、后端、RSSHub 和 QwenPaw 共用的认证密码。 |
| `WEB_HOST_PORT` | 根 `.env` | Web 容器暴露给宿主机的端口，默认 `3080`。 |
| `SECRET_KEY`、`JWT_SECRET_KEY`、`INTERNAL_CALL_SECRET` | `backend/.env` | 应用、JWT 与内部调用安全密钥。 |
| `AI_IDE_LLM_API_KEY` / `LLM_API_KEY` | `backend/.env` | AI 策略与 AI-IDE 的 LLM Key。 |
| `AI_STRATEGY_LLM_REQUIRED` | `backend/.env` | 设为 `true` 时，缺少 LLM Key 会拒绝 Engine 启动；默认 `false`。 |
| `ADMIN_DASHBOARD_*_ENABLED` | `backend/.env` | 可选组件是否纳入后台健康评分。 |

RSSHub 的 Redis URL 默认使用 `REDIS_PASSWORD`。若密码含 URI 保留字符（如 `@`、`:`、`/`），请在根 `.env` 显式填写 URL 编码后的 `RSSHUB_REDIS_URL`。

## 镜像构建与后端代码更新

`quantmind`、`celery-worker` 与 `celery-beat` 共用 `quantmind-oss:latest` 镜像。后端、`config`、`scripts`、策略模板、AlphaAgent 和 TradingAgents-Astock 在构建时复制进镜像；运行时只挂载数据、模型、日志、数据库缓存和用户股票池等可变数据。

因此修改 `backend/` 后，**仅执行 `docker compose restart quantmind` 不会加载新代码**。请构建新镜像并重建使用它的服务：

```bash
docker compose build quantmind
docker compose up -d --force-recreate quantmind celery-worker celery-beat
```

Dockerfile 会先复制并安装 requirements，再复制业务源码。因此只改 `backend/` 时会复用依赖层缓存，不会重新下载所有 Python 包。以下情况才会导致依赖层重新执行：修改 requirements 或其之前的 Dockerfile 层、传入 `--no-cache`、清理构建缓存，或在新机器首次构建。

## 前后端组合与兼容性

详细命令和故障排查见 [前后端容器组合测试](docs/前后端容器组合测试.md)。支持的组合如下：

| 场景 | 后端 | 前端 | 关键处理 |
| --- | --- | --- | --- |
| 1. 镜像验证/部署 | Compose 容器 | Nginx Web 容器 | Nginx 默认代理到 `quantmind:8000`。 |
| 2. 后端本地调试 | 本地 Python | Nginx Web 容器 | 一次性传入 `WEB_API_UPSTREAM=host.docker.internal:8000`，并以 `--no-deps` 防止重启后端容器。 |
| 3. 前端开发 | Compose 容器 | Vite 开发服务器 | 通过同源 `/api`、`/ws` 代理；用 `VITE_API_URL` 指定实际后端地址。 |

```bash
# 场景 2：本地后端 + 容器化 Web
docker compose stop quantmind celery-worker celery-beat
conda run --no-capture-output -n quantmind python -m backend.main_oss

# 另一个终端：不要让 web 的 depends_on 拉起 quantmind
WEB_API_UPSTREAM=host.docker.internal:8000 \
  docker compose up -d --build --force-recreate --no-deps web
```

前端调整包括：

- Web 镜像启动时仅替换 `WEB_API_UPSTREAM`，保留 Nginx 的 `$host`、`$uri` 等原生变量。
- Vite 默认使用相对 `/api` 与 `/ws`，局域网浏览器不会把 API 错误指向访问者自己的 `localhost`。
- Electron 打包版的 `file://` 渲染器仍直连 `http://localhost:8000`，后端 CORS 允许 `null` origin。
- 已保存的旧 Web 端 `:3080` 地址在开发模式会还原为当前 Vite 同源地址，避免开发请求绕过代理。

## Docker-in-Docker 与训练/策略任务

AI-IDE 沙箱、AlphaAgent、RD-Agent 和训练编排会通过 Docker socket 启动子容器。容器内的 `/app/...` 并不是 Docker daemon 所理解的宿主机路径，过去依赖手工设置 `HOST_PROJECT_PATH`，容易在不同部署目录和 Docker Desktop 上失效。

现在 `backend.shared.host_paths` 会读取当前容器的 `/data`、`/app/db`、`/app/models`、`/app/logs` 或 `/app/user_pools_local` bind mount，推导 daemon 可见的项目根目录；宿主机本地运行则直接使用仓库根目录。该逻辑保留 Windows 路径分隔符，并在无法安全解析时明确失败，而非创建错误挂载。

## 数据、存储和服务兼容性

- **数据库启动兼容迁移**：新增的幂等迁移会补齐行情日汇总的来源维度与聚合字段，并将旧订单枚举转换为 API/ORM 使用的小写值及明确的开平仓动作值。新建数据库的初始化 SQL 同步采用新结构。
- **策略存储**：OSS 本地对象存储可以复用对象读取接口；对象存储不可用时回退数据库模式。删除操作会根据实际影响行数返回结果，列表对无效用户 ID 安全返回空列表。
- **行情推送**：默认只使用 Remote Redis；`STREAM_ENABLE_OPENTDX_FALLBACK=true` 时才启用可选的 OpenTDX 直连回退，避免未安装可选依赖时影响主链路。
- **管理后台**：数据库和 Redis 健康检查使用当前运行环境地址及密码；未部署的 data-gateway、QwenPaw、RSSHub、Huntly、Dashboard、Celery Beat 可标为“未启用”，不再使整体健康度降级。
- **LLM 可选化**：未配置 LLM Key 时保留数据、回测、训练和策略等非 AI 能力；AI 策略功能会给出提示。可通过 `AI_STRATEGY_LLM_REQUIRED=true` 恢复严格启动校验。
- **安全清理**：移除了认证失败路径中记录明文凭据的调试日志。

## 验证建议

```bash
# 环境拓扑和宿主机路径单元测试
python -m unittest \
  backend.shared.tests.test_env_loader \
  backend.shared.tests.test_host_paths

# 后端测试（按项目测试入口）
python backend/run_tests.py unit

# 前端类型检查
cd electron && npm run typecheck
```

运行容器化场景后，可使用以下命令检查入口：

```bash
docker compose ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:3080/health
```

## 相关文档

- [旧版 README：完整功能、数据与历史部署说明](docs/README_旧版.md)
- [前后端容器组合测试](docs/前后端容器组合测试.md)
- [本地前后端开发环境启动指南](docs/本地前后端开发环境启动指南.md)
- [后端说明](backend/README.md)
- [部署说明](deploy/README.md)

## 变更边界

本文描述的是当前 `dev` 分支围绕运行环境、配置和兼容性的改造。旧版 README 中的平台功能描述仍然有效；如两份文档的启动命令不一致，以本文和组合测试文档为准。
