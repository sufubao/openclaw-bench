# OpenClaw Bench

`openclaw-bench` 是一个面向 OpenAI 兼容聊天补全服务的确定性基准测试工具，围绕 OpenClaw 的使用模式设计：

- 固定的系统提示词（system prompt）
- 多轮用户对话
- 随机用户到达（而非固定并发数）
- 基于 token 的会话增长和停止规则
- 流式延迟测量（TTFT 和 TPOT）

模拟器始终调用 `POST /v1/chat/completions`。为了让不同轮次的结果可比较，后续轮次使用预生成的占位文本替代模型的实际输出，而非实时模型响应。当不同服务共享同一个配置 UUID 时，结果可直接对比。

## 设计说明

- `users_per_minute` 是到达率（arrival rate），而非并发数。真实并发由到达率、服务延迟和用户思考时间共同决定。
- `think_time_seconds` 是一个分布，而非固定延迟。每轮从配置的分布中采样，具体采样值存储在生成的配置 JSON 中。
- 所有负载计量均基于 token，默认使用从 `request.model` 解析的真实分词器（tokenizer）。不使用字符数来做停止判断。
- 当预估的 prompt token 数超过 `context_threshold * max_context_tokens`，或 prompt 加上下一轮计划分配的 assistant 预算超过硬性上下文限制时，会话将在发送下一个请求之前停止。

## 项目结构

- `generate-config`：读取控制配置 JSON，生成确定性的模拟配置，附带唯一的配置 UUID。
- `simulate`：将配置回放到 OpenAI 兼容的端点，写入包含原始请求指标和聚合摘要的结果 JSON。
- `serve-dashboard`：启动本地 FastAPI 仪表盘，支持上传结果 JSON 并按配置 UUID 分组对比不同运行。

## 快速开始

以可编辑模式安装：

```bash
python -m pip install -e .
```

生成确定性负载配置：

```bash
openclaw-bench generate-config \
  --control-file examples/control.example.json \
  --output out/config.json
```

针对本地 vLLM 服务运行基准测试。`simulate` 现在默认使用 `http://localhost:8000`，因此 `--base-url` 可省略：

```bash
openclaw-bench simulate \
  --config out/config.json \
  --output out/result-server-a.json \
  --run-label run-a \
  --server-label vllm-local
```

如果要对 OpenRouter 或其他托管的 OpenAI 兼容网关做压测，可以覆盖 `--base-url`，并在服务需要时提供 API key：

```bash
export OPENROUTER_API_KEY=your_key_here

openclaw-bench simulate \
  --config out/config.json \
  --output out/result-server-a.json \
  --base-url https://openrouter.ai/api \
  --run-label run-a \
  --server-label router-a \
  --http-referer https://your-benchmark-host.example \
  --title openclaw-bench
```

启动对比仪表盘：

```bash
openclaw-bench serve-dashboard --host 127.0.0.1 --port 8000
```

然后访问 `http://127.0.0.1:8000`，上传多个结果 JSON 文件，即可对比共享同一配置 UUID 的不同运行。

## 控制配置文件

以 [`examples/control.example.json`](examples/control.example.json) 作为起点。

重要字段说明：

- `total_users`：要模拟的总会话数。
- `arrival.users_per_minute`：会话到达率（每分钟到达的用户数）。
- `think_time_seconds`：用户收到回复后的思考时间分布。
- `initial_user_tokens`：首轮用户输入的 token 大小。
- `context_increment_tokens`：后续轮次用户输入的 token 大小。
- `assistant_tokens`：确定性 assistant 占位文本大小，同时也是实际请求的 `max_tokens` 上限。
- `max_context_tokens`：模型硬性上下文窗口大小，例如 `262144` 表示 256K 目标。
- `context_threshold`：当 prompt 预估 token 数达到硬性限制的该比例时，会话停止发送后续请求。

支持的数值分布：

- `constant` — 常数
- `uniform` — 均匀分布
- `triangular` — 三角分布
- `lognormal` — 对数正态分布

支持的分词器模式：

- `model`：默认模式。从 `request.model` 解析分词器，已知模型族使用 `tiktoken`，其他使用 `transformers` 加载。
- `tiktoken`：显式指定 `tiktoken` 模式，强制使用已知的编码族。
- `tokenizer_file`：加载本地 `tokenizer.json` 文件，无需远程注册表查找。
- `regex`：离线正则回退模式，仅用于冒烟测试，不建议用于正式基准测试。

## 输出指标

每个结果 JSON 包含：

- 每个请求的原始计时和 token 计数
- 每个会话的完成/失败计数
- 聚合吞吐量（请求数/秒、输出 token 数/秒）
- TTFT 百分位数：`p5`、`p10`、`p50`、`p90`、`p99`
- TPOT 百分位数：`p5`、`p10`、`p50`、`p90`、`p99`
- 总延迟百分位数
- 峰值在途请求数（peak in-flight request count）

## 验证

运行单元测试：

```bash
pytest
```

## API 说明

模拟器针对 OpenAI 兼容的聊天补全接口编写，兼容 OpenRouter 的 `/v1/chat/completions` 路径。OpenRouter 当前文档表明，用量信息会自动包含在响应中（包括流式响应）：

- https://openrouter.ai/docs/guides/guides/administration/usage-accounting
