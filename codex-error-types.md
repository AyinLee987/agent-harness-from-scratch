# Codex 错误系统总结

> 本文基于 OpenAI Codex CLI（codex-rs）源码梳理，聚焦「错误类型定义、分类机制、错误生命周期、无限循环控制」四块内容。
> 所有文件路径相对于 `codex-rs/` 目录。

---

## 1. 总览：三个正交的「分类」轴

Codex 里没有一个大统一的「可恢复分类器」。所谓「错误分类」被拆成了三个正交的问题，各自在不同层、用不同方式回答：

| 层级 | 问的是 | 判定方式 | 位置 |
|---|---|---|---|
| 工具层 | 模型换动作能修复吗？ | 开发者编译期手动 `map_err` | 各 handler |
| 沙箱层 | 命令失败是沙箱造成的吗？ | 运行时启发式（关键词 + 退出码） | `sandboxing/src/denial.rs` |
| 传输层 | 这个 HTTP 请求该重发吗？ | 枚举白名单 `is_retryable()` | `protocol/src/error.rs` |

三层互不相干，下面分别展开。

---

## 2. 核心错误类型定义

### 2.1 `FunctionCallError` — 工具错误的唯一分类

`tools/src/function_call_error.rs`：

```rust
#[derive(Debug, Error, PartialEq)]
pub enum FunctionCallError {
    #[error("{0}")]
    RespondToModel(String),   // 可恢复：错误文本回传给模型
    #[error("Fatal error: {0}")]
    Fatal(String),            // 不可恢复：中止会话
}
```

- **只有两个变体**，没有第三种。
- **没有任何 `From<anyhow::Error>` 之类的 blanket 转换** —— 这是关键。它意味着 Rust 编译器强迫每个错误路径都必须被显式归类成两个变体之一，否则编译不过。

### 2.2 `CodexErrorDetails` — 会话级错误分类

`protocol/src/error.rs`。覆盖整个会话/驱动的错误类别，比 `FunctionCallError` 宽得多：

```
TurnAborted, SessionBudgetExceeded, Stream, ContextWindowExceeded,
Timeout, Interrupted, InvalidRequest, Sandbox, UnsupportedOperation,
Fatal, Io, Json, TokioJoin, InternalAgentDied, ConnectionFailed,
InternalServerError, RequestTimeout, ...
```

### 2.3 `SandboxErr` — 沙箱错误

`protocol/src/error.rs:36`：

```
Denied / SeccompInstall / SeccompBackend / Timeout / Signal / LandlockRestrict
```

---

## 3. 分类机制详解

### 3.1 工具层：编译期手动分类（主要）

**没有运行时分类器。** 分类就是代码里每个 `map_err` 处的人工选择：

```rust
.map_err(FunctionCallError::RespondToModel)?;   // 可恢复
.map_err(FunctionCallError::Fatal)?;            // 不可恢复
```

开发者判据就一条：**「模型换一个动作，能不能解决？」**

- **能 → `RespondToModel`**（错误文本回给模型）
- **不能 → `Fatal`**（中止会话）

**分类的粒度是「操作（代码位置）」，不是「错误值」。** 一个 `map_err` 覆盖一整类错误：

```rust
let output = run_command(&cmd).await
    .map_err(FunctionCallError::RespondToModel)?;
```

这一处 `map_err` 把 `run_command` 的**所有**失败（命令不存在、权限不够、段错误、超时、OOM…）都归成 `RespondToModel`。开发者不需要枚举每个具体错误，只需要在每个「可能失败的操作」处放一个闸门。

**决定分类的是「操作的性质」，不是「操作的类型」本身。** 同样「读文件」，读工具自己的配置文件失败 → `Fatal`；读用户要看的文件失败 → `RespondToModel`。

两边典型例子：

| 归类 | 例子 | 位置（示例） | 为什么 |
|---|---|---|---|
| `RespondToModel` | 参数解析失败 | 各 handler | 模型重发一份对的 |
| `RespondToModel` | 命令执行失败/超时 | shell handler | 模型换命令重试 |
| `RespondToModel` | apply_patch 被安全规则拒 | apply_patch | 模型改补丁 |
| `RespondToModel` | 网络被策略拦 | 网络 handler | 模型换主机/申请权限 |
| `RespondToModel` | `UnsupportedOperation` | `core/src/agent/agent_resolver.rs:26` | 显式映射 |
| `Fatal` | `"failed to read current time"` | `current_time.rs:99` | 系统时钟坏了 |
| `Fatal` | `"failed to sleep"` | `sleep.rs:123` | 模型改不了 |
| `Fatal` | `"failed to load rules"` | `session/mod.rs:592` | 环境问题 |
| `Fatal` | `"tool produced no output"` | `core/src/tools/registry.rs:685` | 工具契约被违反 |
| `Fatal` | 线程状态损坏 / 计数溢出 | `codex_thread.rs:725` | 内部不变量打破 |

### 3.2 沙箱层：唯一真正的运行时启发式

`is_likely_sandbox_denied()` 在 `sandboxing/src/denial.rs:13`，是**唯一一处运行时读错误内容做分类**的代码。它回答一个更窄的问题：**「这条命令失败，是不是沙箱拦的？」** 决定是否触发「无沙箱重试」。

```rust
pub fn is_likely_sandbox_denied(sandbox_type, exec_output) -> bool {
    if sandbox_type == None || exit_code == 0 { return false; }   // 没沙箱/成功 → 不是

    // ① 关键词匹配：输出里出现这些词 → 判成沙箱拒了（大小写不敏感）
    ["operation not permitted", "permission denied", "read-only file system",
     "seccomp", "sandbox", "landlock", "failed to write file"]
        .iter().any(|kw| output.contains(kw));

    // ② 快速排除：2/126/127 是命令自己的错，不是沙箱
    if [2, 126, 127].contains(&exit_code) { return false; }

    // ③ Linux seccomp 特征：exit == 128 + SIGSYS → 沙箱杀了它
    if sandbox == LinuxSeccomp && exit_code == 128 + SIGSYS { return true; }
}
```

注释自己写明：*"We don't have a fully deterministic way to tell if our command failed because of the sandbox"* —— 是保守猜测。

判定为沙箱拒绝后，`core/src/tools/orchestrator.rs` 会在用户批准后做一次「无沙箱重试」，重试理由文案为 `"command failed; retry without sandbox?"`（`orchestrator.rs:527`）。

### 3.3 传输层：`is_retryable()` — 与工具错误无关

`protocol/src/error.rs:359`。它分类的是**「这个模型 HTTP 请求要不要重发一次」**（网络瞬态错误），跟「工具错误喂不喂给模型」是两条正交的轴：

```rust
// 可重试（瞬态）：Stream / Timeout / RequestTimeout / ConnectionFailed /
//                InternalServerError / Io / Json / TokioJoin / InternalAgentDied
// 不可重试：TurnAborted / Fatal / QuotaExceeded / ContextWindowExceeded /
//           Sandbox(_) / InvalidRequest / UnsupportedOperation ...
```

注意 `Sandbox(_)` 在「不可重试」里——沙箱拒绝重发同样的 HTTP 请求没用，得换内容，这和工具层的「无沙箱重试」是两码事。

---

## 4. 错误的生命周期（从产生到进 prompt）

```
① 工具执行失败（运行时）
        ↓
② 错误值落在本地 Rust 代码手里
        ↓
③ 代码按预先写死的规则归类（RespondToModel / Fatal）
        ↓
④ RespondToModel → 包成 FunctionCallOutput 消息写入对话历史
   Fatal         → return Err(CodexErr::Fatal) 中止会话
        ↓
⑤ needs_follow_up = true → turn 循环再走一轮
        ↓
⑥ 下一轮调模型时，对话历史整体序列化成请求 input，错误随行进 prompt
```

关键代码位置：

- `core/src/stream_events_utils.rs:362`（RespondToModel 分支）：

```rust
Err(FunctionCallError::RespondToModel(message)) => {
    let response = ResponseInputItem::FunctionCallOutput {
        call_id: String::new(),          // ← 注意是空的
        output: FunctionCallOutputPayload {
            body: FunctionCallOutputBody::Text(message),
            ..Default::default()
        },
    };
    // ... record_conversation_items 写入对话历史
    output.needs_follow_up = true;
}
```

- `core/src/stream_events_utils.rs:384`（Fatal 分支）：

```rust
Err(FunctionCallError::Fatal(message)) => {
    return Err(CodexErr::Fatal(message));
}
```

**要点：**

1. **错误没有专门的「错误槽」**。它进 prompt 的方式和「工具成功返回输出」完全相同——就是一条 `FunctionCallOutput` 消息。模型看到的是「这个工具返回了这样一段文本」，至于是错误还是成功，是模型自己读出来的，prompt 层不区分。
2. **`call_id` 是空的**（`String::new()`）。正常工具结果带 `call_id` 好让模型把「调用」和「结果」对上；但 `RespondToModel` 这条错误消息不带 call_id，只是孤零零一句「这里有问题，你继续」，配合 `needs_follow_up` 让模型再走一轮。
3. **分类不进 prompt**。进 prompt 的是那段**错误文本**；「可恢复/致命」这个分类只决定文本有没有资格进 prompt。

---

## 5. 未分类的错误（panic）

「没编码过」的错误分两种，结局完全不同：

### 5.1 `Result` 错误但忘了 `.map_err`

**编译期就被拦下。** 因为 `FunctionCallError` 没有 blanket `From` 转换，在一个返回 `Result<_, FunctionCallError>` 的函数里用 `?` 传播非 `FunctionCallError` 错误，Rust 编译器直接报「类型不匹配」，拒绝编译。所以「运行时遇到一个没写分类规则的错误」这种情况物理上不存在。

### 5.2 运行时 panic（`.unwrap()` / `.expect()` / `panic!` / 越界 / 溢出）

这才是真正的「没编码过的错误」——失败模式压根没被建模成 `Result`。

- 主 dispatch 路径 `core/src/tools/registry.rs:741` 的 `tool.handle(...).await?` 周围**没有 `catch_unwind`**。
- panic 一路 unwind，tokio 在 task 边界截住，最终作为致命错误冒泡到顶层，会话崩掉。
- **panic 不会被转成 `RespondToModel`**——模型看不到、也没机会恢复。
- 唯一例外：`code-mode-runtime` 的 V8 子系统有显式 `catch_unwind`（`cell_actor/callbacks.rs:55`、`runtime/mod.rs:129`），但也是转成一段错误文本 + 失败上报，不是可恢复的 `RespondToModel`。

---

## 6. 无限循环控制

**关键结论：Codex 没有「轮次上限」，也没有「检测到重复动作就打断」的循环检测器。** 它靠间接机制，核心是「预算」当兜底。

| 机制 | 类型 | 作用 | 位置 |
|---|---|---|---|
| Token 预算 `SessionBudgetExceeded` | **硬兜底** | 耗尽即中止，无限循环不可能 | `core/src/session/rollout_budget.rs:33` |
| 提醒消息 | 软约束 | 预算快完时催模型收尾 | `core/src/session/token_budget.rs:66` |
| 自动压缩 | 让循环「可持续」 | 腾上下文，但不打断循环 | `core/src/session/turn.rs:419-458` |
| 单工具超时 | 防卡死 | 单个命令执行超时就杀（默认 10s） | `core/src/exec.rs` |
| 轮次上限 / 循环检测 | **不存在** | —— | —— |

### 6.1 Token 预算 = 唯一硬上限

```rust
// core/src/session/rollout_budget.rs:26
pub(crate) fn record_rollout_budget_usage(&self, usage: &TokenUsage) -> CodexResult<()> {
    if self.services.agent_control.rollout_budget().record_usage(usage)? {
        return Err(CodexErr::SessionBudgetExceeded);
    }
    Ok(())
}
```

模型每消耗一轮 token 就在这里记账；记到超了直接 `SessionBudgetExceeded` 中止。所以「无限循环」物理上不可能——预算有穷，耗尽就停。

### 6.2 提醒消息 = 软性催促

`token_budget.rs:66` 的 `maybe_record` 两级：

```rust
// ① 剩余 token 跌破阈值 → 注入「快没预算了，请收尾」提醒
if base_window_tokens_remaining <= reminder_threshold_tokens { ... }

// ② 剩余 = 0 → 注入 auto_compact_fallback_prompt（强制收尾提示）
if base_window_tokens_remaining == 0 { ... }
```

这些是**软约束**：往对话里塞一句话让模型自己意识到该结束，不是硬停。

### 6.3 自动压缩 = 让循环在预算内继续（但不断它）

```rust
// core/src/session/turn.rs:419
let should_roll_over = needs_follow_up
    && (sess.take_new_context_window_request().await || token_limit_reached);

if should_roll_over {
    run_auto_compact(...).await;   // 摘要压缩历史，腾出上下文
    continue;                      // 继续下一轮
}
```

上下文快满时把历史摘要压缩腾空间。关键注释（`turn.rs:430`）道出设计哲学：

```rust
// as long as compaction works well in getting us way below the token limit,
// we shouldn't worry about being in an infinite loop.
```

**Codex 不检测循环，只保证「预算」这面有穷的墙始终存在**：循环想跑多久都行，预算一耗尽就 `SessionBudgetExceeded` 停掉。

---

## 7. 关键文件速查表

| 文件（相对 `codex-rs/`） | 内容 |
|---|---|
| `tools/src/function_call_error.rs` | `FunctionCallError` 枚举（RespondToModel/Fatal） |
| `tools/src/tool_executor.rs` | `ToolExecutor::handle` 返回 `Result<Box<dyn ToolOutput>, FunctionCallError>` |
| `protocol/src/error.rs` | `CodexErrorDetails`、`SandboxErr`、`is_retryable()` |
| `sandboxing/src/denial.rs` | `is_likely_sandbox_denied()` 运行时启发式 |
| `core/src/stream_events_utils.rs` | RespondToModel/Fatal 的处置（362/384） |
| `core/src/tools/registry.rs` | 工具分发 + 防御性 `Fatal("tool produced no output")`（685）、`handle_any_tool`（741） |
| `core/src/tools/orchestrator.rs` | 沙箱拒绝后的无沙箱重试（527） |
| `core/src/agent/agent_resolver.rs` | `UnsupportedOperation` → `RespondToModel` |
| `core/src/session/rollout_budget.rs` | `SessionBudgetExceeded` 硬兜底 |
| `core/src/session/token_budget.rs` | 提醒消息注入 |
| `core/src/session/turn.rs` | turn 循环、`should_roll_over`、自动压缩 |
| `core/src/exec.rs` | `DEFAULT_EXEC_COMMAND_TIMEOUT_MS = 10_000` |
| `code-mode-runtime/src/cell_actor/callbacks.rs` | code-mode 的 `catch_unwind` 兜底 |

---

## 8. 一句话总结

> 错误是「先本地接住、本地归类，再决定要不要进模型视野」。归类由**操作的性质**（编译期写死）决定，错误文本由**实际失败**（运行时）决定。没有统一的运行时分类器——工具层靠人工 `map_err`，沙箱层靠关键词启发式，传输层靠枚举白名单。无限循环靠**预算墙**兜底，而非检测器。
