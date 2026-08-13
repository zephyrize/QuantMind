# QuantMind Docker 环境重启与更新指南

本文适用于项目根目录的 `docker-compose.yml`（OSS 单镜像后端部署）。所有命令均在项目根目录执行，并使用 Docker Compose V2：

```powershell
docker compose <命令>
```

## 核心规则

| 变更类型 | 是否需要重建镜像 | 推荐操作 |
| --- | --- | --- |
| `backend/`、`config/`、`scripts/`、策略模板等已挂载的后端源码 | 否 | 重启使用该代码的容器 |
| `electron/` 前端源码 | 是 | 重新构建并重建 `web` |
| `data-gateway` 所用后端源码 | 是 | 重新构建并重建 `data-gateway` |
| `dashboard/` 页面代码 | 否 | 重启 `dashboard` |
| Python / Node 依赖清单或对应 Dockerfile | 是 | 构建受影响服务的镜像并重建容器 |
| `.env` 中被 Compose 注入的变量 | 否 | 用 `up -d --force-recreate` 重建受影响容器 |

`restart` 只重启既有容器和进程：不会读取新的 Compose 环境变量，也不会构建新镜像。`up -d --force-recreate` 会依照当前 Compose 配置和 `.env` 重建容器；加入 `--build` 后还会先构建镜像。

> [!WARNING]
> 不要把 `docker compose up -d --build --force-recreate` 当作日常更新命令。它会尝试构建 Compose 中**所有声明了 `build:` 的本地镜像**，包括 `quantmind`、`web`、`data-gateway` 和 `dashboard`。若 Docker 构建缓存失效，可能重新下载并安装大量 Python 或 Node 依赖，耗时很长。

日常更新应始终指定目标服务；前端 Web 的最小影响命令见下文“Web 前端源码”。

## 服务与容器对应关系

| Compose service | 容器名 | 用途 |
| --- | --- | --- |
| `db` | `quantmind-db` | PostgreSQL |
| `redis` | `quantmind-redis` | Redis |
| `quantmind` | `quantmind` | API（8000）、Engine（8001）、Trade（8002）、Stream（8003） |
| `celery-worker` | `quantmind-celery` | Qlib 回测异步任务 |
| `celery-beat` | `quantmind-celery-beat` | 定时任务调度 |
| `web` | `quantmind-web` | React/Vite 构建后的 Nginx Web 前端 |
| `data-gateway` | `quantmind-data-gateway` | 多数据源网关（8004） |
| `dashboard` | `quantmind-dashboard` | OpenBB / Streamlit 看板 |
| `huntly` | `quantmind-huntly` | 财经资讯聚合 |
| `rsshub` | `quantmind-rsshub` | RSS 服务 |
| `qwenpaw` | `qwenpaw` | QuantBot / QwenPaw |

## 按改动类型更新

### 1. 后端源码、配置目录或策略模板

`quantmind`、`celery-worker` 和 `celery-beat` 均把宿主机的 `backend/`、`config/`、`scripts/` 等目录挂载到容器中。因此文件一经保存即可在容器内看到；但 Python 进程不会自动热重载，需重启相应服务。

```powershell
# 通常的后端改动：主服务和所有异步任务一起更新
docker compose restart quantmind celery-worker celery-beat

# 仅 API / Engine / Trade / Stream 主服务改动
docker compose restart quantmind

# 仅 Qlib 回测 worker 改动
docker compose restart celery-worker

# 仅定时任务改动
docker compose restart celery-beat
```

无需重新安装 Python 依赖，也无需重新构建 `quantmind-oss` 镜像。

### 2. Web 前端源码（`electron/`）

`web` 镜像在构建阶段执行 Vite build，并将静态产物复制到 Nginx；宿主机前端源码没有挂载进该容器。修改 `electron/`、`scripts/frontend/` 或 Web Nginx 配置后执行：

推荐拆成两步执行，以确保只构建和替换 `web`，不处理 `quantmind` 这个依赖服务：

```powershell
# 只构建 Web 前端镜像（仅应看到 npm / Vite 构建步骤）
docker compose build web

# 只重建 Web 容器；--no-deps 禁止 Compose 启动、重建或构建后端依赖
docker compose up -d --no-deps --force-recreate web
```

前提是 `quantmind` 已正常运行。若后端也需要更新，请单独按后端改动的命令处理。

`docker compose up -d --build web` 在理论上应构建 `web` 镜像；但 `web` 依赖 `quantmind` 的健康状态，实际执行环境、Compose 版本或构建缓存状态可能使其同时处理依赖服务。若输出出现 `pip install`、`fastapi`、`pandas`、`akshare`、`celery` 等 Python 包，说明已经进入后端 `quantmind-oss` 的构建流程；这不是前端依赖。使用上面的 `build web` 加 `up --no-deps` 可以避免这类连带操作。

仅改前端业务代码时也需要重新 build，但 Docker 通常会复用依赖安装层；通常不需要重新下载或安装 Node 依赖。若改动 `electron/package.json` 或锁文件，则构建会重新安装前端依赖。

`docker/Dockerfile.web` 中只有 `npm install` 和 Vite build，没有 `pip install`。因此下列包属于后端依赖而不是前端依赖：`fastapi`、`pandas`、`akshare`、`celery`、`sqlalchemy`、`uvicorn` 等。

> 本地 Vite/Electron 开发模式不使用 `web` 容器；可在项目根目录运行 `npm run dev:web`，Vite 会自动热更新页面。

### 3. Data Gateway

`data-gateway` 的代码由 Dockerfile `COPY backend/` 写入镜像，运行时没有源码挂载。修改网关相关后端代码、`docker/Dockerfile.data-gateway` 或其依赖后执行：

```powershell
docker compose up -d --build --force-recreate data-gateway
```

### 4. Dashboard

`dashboard/`、`data/`、`db/` 在 `dashboard` 容器中是挂载卷。修改 Streamlit 页面代码后：

```powershell
docker compose restart dashboard
```

若修改 `dashboard/requirements.txt` 或 `docker/Dockerfile.dashboard`，则改为：

```powershell
docker compose up -d --build --force-recreate dashboard
```

### 5. 修改 `.env`

Compose 会在创建容器时读取 `.env`，并把其中的值写入容器环境变量。因此 DB 连接、端口、LLM Key、`WEB_PORT`、`HUNTLY_*`、`QUANTDB_API_KEY` 等变量变更后，必须重建使用这些变量的容器：

```powershell
# 常用后端配置变更
docker compose up -d --force-recreate quantmind celery-worker celery-beat

# 同时刷新所有可能读取 .env 的应用服务
docker compose up -d --force-recreate quantmind celery-worker celery-beat dashboard huntly rsshub qwenpaw web data-gateway
```

根目录 `.env` 还以只读方式挂载到 `quantmind:/app/.env`。管理员账号配置（`ADMIN_USERNAME`、`ADMIN_EMAIL`、`ADMIN_PASSWORD`）在主服务启动时直接读取该文件，故仅修改这些值时可使用：

```powershell
docker compose restart quantmind
```

这会在启动时同步管理员账号；其他由 Compose 注入的 `.env` 变量仍应使用 `--force-recreate`。

### 6. 修改依赖或 Dockerfile

| 修改位置 | 命令 |
| --- | --- |
| `requirements.txt`、`requirements/production.txt`、`requirements/ai.txt`、`docker/Dockerfile.oss` | `docker compose build quantmind`，再 `docker compose up -d --force-recreate quantmind celery-worker celery-beat` |
| `electron/package.json`、前端锁文件、`docker/Dockerfile.web` | `docker compose up -d --build --force-recreate web` |
| `docker/Dockerfile.data-gateway` 或网关依赖 | `docker compose up -d --build --force-recreate data-gateway` |
| `dashboard/requirements.txt`、`docker/Dockerfile.dashboard` | `docker compose up -d --build --force-recreate dashboard` |

一般无需手工执行 `pip install` 或 `npm install`：它们由对应 Dockerfile 的构建步骤执行。仅当依赖清单或基础镜像发生变化时，Docker 才会重新执行相关安装层。

Python 镜像的 Dockerfile 使用 `pip --no-cache-dir`，因此它不保留 pip 下载缓存；不过只要 Docker 的**镜像层缓存**仍在且依赖层之前的输入没有变化，Docker 会直接复用整个依赖安装层，不会再次执行 pip。出现完整的 `Installing collected packages` 通常表示该镜像层缓存已经失效或被清理，常见原因包括：修改了 requirements、Dockerfile 中依赖安装前的内容或构建参数，切换 BuildKit builder / Docker daemon，或执行过 Docker Desktop 清理、`docker builder prune`、`--no-cache`。

## 所有服务的重启命令

```powershell
docker compose restart db
docker compose restart redis
docker compose restart quantmind
docker compose restart celery-worker
docker compose restart celery-beat
docker compose restart web
docker compose restart data-gateway
docker compose restart dashboard
docker compose restart huntly
docker compose restart rsshub
docker compose restart qwenpaw
```

重启全部运行中的服务：

```powershell
docker compose restart
```

按当前 Compose 文件和 `.env` 重建全部容器（不构建镜像）：

```powershell
docker compose up -d --force-recreate
```

构建全部本地镜像并重建全部容器：

```powershell
# 高影响操作：会构建 quantmind、web、data-gateway、dashboard 等所有本地镜像。
# 若任一构建缓存失效，可能重新安装大量 Python / Node 依赖；仅在完整发布或排障时使用。
docker compose up -d --build --force-recreate
```

> [!CAUTION]
> 上述“构建全部本地镜像”命令不是常规前端更新命令。前端日常更新请使用 `docker compose build web` 和 `docker compose up -d --no-deps --force-recreate web`。

## 检查状态与日志

```powershell
# 查看容器状态与健康检查结果
docker compose ps

# 查看所有服务最近日志
docker compose logs --tail 200

# 持续跟踪某个服务日志
docker compose logs -f quantmind
docker compose logs -f celery-worker
docker compose logs -f web
```

## 数据库与 Redis 注意事项

`db` 使用命名卷 `postgres-data`，`redis` 使用命名卷 `redis-data`。普通 `restart`、`up -d --force-recreate` 或镜像重建都不会删除它们的数据。

`POSTGRES_DB`、`POSTGRES_USER`、`POSTGRES_PASSWORD` 只在 PostgreSQL 数据目录首次初始化时用于创建数据库和初始账号。修改这些 `.env` 值并重建 `db` 不会自动修改已有数据库的账号、密码或数据库名。除非确认可以丢弃数据并已备份，否则不要为此删除 `postgres-data` 卷。
