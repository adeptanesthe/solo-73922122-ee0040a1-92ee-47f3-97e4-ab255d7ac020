# 片库镜头检索向量服务

用可播镜头实时维护视频检索向量：导入按时间排序的 1～100,000 个镜头（每个带 16 维特征、注意力分数），
按单号登记/撤销镜头隐藏区间，每次变更后以 **O(log N)** 返回由可播镜头算出的检索向量
（注意力分数 softmax 加权特征和）。

另提供**定位 API**：给定 16 维查询向量与所需正贡献占比，返回镜头数最少（同长取最早）的
连续**可播**区间，其段内正负贡献之和达到全片正贡献之和的指定比例。

## 运行

```bash
# Docker Compose（推荐）
docker compose up --build

# 或本地运行
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务监听 `:8000`，健康检查 `GET /health`，交互文档 `GET /docs`。

演示脚本（重叠区间的登记与撤销，含幂等重送、冲突、全隐藏撤下、恢复）：

```bash
bash scripts/demo.sh            # BASE=http://localhost:8000 可覆盖
```

定位演示（负贡献如何改变选段、隐藏墙切分可播区间、两种无结果状态）：

```bash
bash scripts/demo_locate.sh
```

自测：

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/videos/{video_id}/shots` | 导入镜头。body: `{"shots":[{"feature":[16 个 −1~1],"score":−1000~1000}, ...]}`，1～100,000 条。重复导入整体替换并清空该视频所有订单 |
| `POST` | `/videos/{video_id}/restrictions` | 登记隐藏单。body: `{"order_id":"lic-A","start":2,"end":6}`，区间 0 起算、左闭右开 |
| `DELETE` | `/videos/{video_id}/restrictions/{order_id}` | 撤销一单；仍被其他单覆盖的镜头保持隐藏 |
| `GET` | `/videos/{video_id}/retrieval-vector` | 查询当前检索向量 |
| `POST` | `/videos/{video_id}/locate` | 定位最短达标连续可播区间。body: `{"query":[16 维有限数],"ratio":>0 且 ≤1}` |
| `GET` | `/health` | 健康检查 |

统一响应：

```json
{"video_id":"demo","shot_count":10,"visible_count":6,"status":"ok","vector":[...16 维...]}
```

定位响应：

```json
{"video_id":"demo","shot_count":8,"visible_count":8,"status":"ok","ratio":0.5,
 "positive_total":0.675,"start":2,"end":5,"achieved_ratio":0.5,
 "message":"定位成功"}
```

- **幂等**：同 `order_id` 同范围重送 → 200，状态不变。
- **冲突**：同 `order_id` 不同范围 → 409（提示先撤销）。
- **全隐藏**：`status="withdrawn"`、`vector=null` —— 即撤下索引的标记；撤销后自动恢复。
- 校验失败 422（维度数、特征/分数越界、镜头数越界、查询含非有限值、ratio 越界），区间语义错误 400（`start>=end`、`end>N`），未知视频/单号 404。

### 定位语义（`/locate`）

对每个可播镜头 i（全片可播集合 V）：

```
w_i = softmax(attention)_i                 # 权重在 V 上求和为 1
c_i = w_i · dot(feature_i, query)          # 带符号贡献，可能为负
positive_total = Σ_{c_i > 0} c_i           # 全片正贡献之和
target = positive_total · ratio
```

区间必须是原镜头下标上的连续段且**段内每个镜头都可播**（隐藏镜头会切断可播段）；
段贡献是段内 `c_i` 的**带符号和**，负贡献会拉低段贡献、迫使选段变长或改址。
在所有段贡献 ≥ target 的区间中返回**镜头数最少**者，同长取**最早**；返回 0 起算、
左闭右开的 `[start, end)` 与实际达到的占比 `achieved_ratio`。

两种无结果状态（HTTP 均为 200，靠 `status` 区分）：

- `no_positive_contribution`：全片无可播镜头，或所有可播镜头对该查询都无正贡献（`positive_total=0`）。
- `no_qualifying_interval`：存在正贡献，但没有任何连续可播区间（含段内负贡献相抵后）能达到所填占比。

**取态语义**：每次定位在进入时取同一份快照（当时的镜头与生效限播单）；定位的 O(N)
扫描在锁外基于不可变快照进行，扫描期间到达的登记/撤销照常生效，但不改变本次定位结果。

## 实现

### 检索向量（`app/segment_tree.py`）

线段树维护可播集合上的稳定 softmax 加权和。每个节点存：

- `cnt`：可播镜头数
- `mx`：可播注意力分数最大值
- `sm = Σ exp(score − mx)`
- `vec[16] = Σ exp(score − mx) · feature`

兄弟合并时平移到共同最大值（经典 online-softmax 分解）：

```
m   = max(mx_l, mx_r)
sm  = sm_l·exp(mx_l−m) + sm_r·exp(mx_r−m)
vec = vec_l·exp(mx_l−m) + vec_r·exp(mx_r−m)
```

隐藏用**懒引用计数**：登记对区间覆盖的 O(log N) 个规范节点 `cover += 1`，撤销 `cover -= 1`；
`cover > 0` 的节点贡献为空。多单重叠时同一镜头计数 ≥1，撤销一单后若仍被其他单覆盖则保持隐藏。
子节点聚合始终有效，无需下推。检索向量 = 根 `vec / sm`（逐维）。

**复杂度**：登记/撤销各 O(log N) 次节点访问（每次合并为 16 维常数工作）；导入建树 O(N)；查询 O(1)。
N=100,000 时树约 2¹⁸ 个节点，扁平 `array('d')` 存储，内存约 40 MB/视频。

### 定位（`app/locate.py`）

1. **覆盖差分数组**：对 K 个生效单做 O(N+K) 的区间加减，线性扫描得到每镜头是否可播。
2. **贡献**：用树根的 softmax 最大值与分母算每镜头权重（与检索向量同一数值路径），再算点积与带符号贡献。
3. **最短达标子段**：对每个极大可播段跑单调队列（LeetCode 862 型）。前缀入队时按严格递增剪枝
   （被支配的前缀不可能产生更短达标窗口），队首在窗口和达标时弹出，每个前缀至多入队/出队一次 → O(N)。

**数值处理（关键）**：窗口和是前缀之差。普通 double 前缀在严重抵消下会丢位——一个 O(1) 前缀上
叠加 1e-12 级的小项已低于 double ulp，任何后续减法（含 Kahan、含对已存前缀再 `math.fsum`）都无法
恢复（已实测验证）。因此每个前缀以 **double-double（hi, lo 两个 double，约 31 位有效十进制数字）**
保存，每次加减仅常数次浮点运算；窗口差用 Knuth two-sum 在 hi/lo 上精确求得。阈值比较只需相对
`positive_total` 的 1e-12 容差吸收贡献本身的几 ulp 舍入，**不设绝对地板**，因此极小 target 不会被容差吞掉。
支配判断也在 double-double 上做（两个不同的 dd 前缀可能共享同一个 double 高位）。

**复杂度**：O(N + K)，N=镜头数、K=生效单数。快照在锁内 O(1+K) 取定，扫描完全在锁外。

**精度**：合并只做 max 平移与浮点乘加，与"稳定全量计算"（max 平移 + `math.fsum` 精确求和的参考实现）
的差异仅来自求和结合顺序，误差随树高 ~log₂N 线性累积，理论量级 ~1e-15，远小于 1e-9 的验收阈值。
分数区间 [−1000, 1000] 内 `exp` 不会上溢，下溢到 0 与全量计算行为一致。

## 实测结果（本机 Python 3.11，pytest 9.2）

- **89 个测试全部通过**（`tests/test_service.py`，原 47 个检索向量测试 + 新增 42 个定位测试）：
  - 40 组随机用例（N=1…1000，每步登记/撤销后比对参考实现）：逐维误差 ≤ 1e-9 ✅
  - 极端值（分数 ±1000、特征 ±1）✅；N=100,000 大样本 ✅；全隐藏→恢复 ✅
  - API：重叠撤销、幂等重送、409 冲突、404、422/400 校验、withdrawn 标记 ✅
  - **定位**：30 组随机用例 + 2000 组极端抵消用例，与 O(N²) `math.fsum` 精确 oracle 完全一致
    （最短优先、同长最早、隐藏墙切分、两种无结果状态）；含权重下溢不得切碎可播段的专项回归 ✅
- **定位数值正确性（独立 oracle 复核）**：另跑 6000 组随机 + 1e14 动态范围极端用例，
  与精确 `fsum` 暴力枚举零不一致；10 万镜头端到端答案与独立 `fsum` 重算逐位相等（diff=0.00e+00）。
- **定位复杂度实测（O(N+K)）**：
  - N=10,000 → 93 ms；N=100,000 → 948 ms（10 倍镜头 ≈ 10 倍耗时，线性；对抗场景下同样严格线性）。
  - 生效单数 K=10 → 470 ms，K=2000 → 347 ms（差分构造成本随 K 增长但被扫描摊薄，无二次项）。
  - 纯 Python + double-double 常数较大（每镜头约十次浮点运算）；10 万镜头单次定位 < 1 s。
- **精度实测**：N=100,000 下 60 个随机隐藏状态，检索向量逐维最大误差 **8.3e-17**（阈值 1e-9）。
- **检索向量复杂度实测**（2000 次登记 + 2000 次撤销取平均）：
  - N=10,000 → 89.7 µs/次；N=100,000 → 181.5 µs/次。N 涨 10 倍耗时仅 ~2 倍，符合 O(log N)。
- **端到端（HTTP，uvicorn 单 worker）**：导入 100,000 镜头（16 MB body）约 5～7 s；
  登记/撤销单次 1.5～10 ms；定位 10 万镜头 0.8～1.0 s；
  各 ratio（0.1/0.5/0.9/1.0）返回段长随占比单调不减，achieved_ratio 均达标。
- **并发取态**：40 次定位与 80 次登记/撤销并发，每次定位基于进入时快照，结果全部自洽
  （区间不越界、达标占比成立），扫描期间到单不阻塞且不影响本次结果。
- **演示**：
  - `scripts/demo.sh`：撤销 lic-A 后 [4,8) 仍被 lic-B 隐藏，全部撤销后向量与导入时逐位一致。
  - `scripts/demo_locate.sh`：负贡献把 ratio=0.5 的最短段从长度 1 推到 [2,5) 长度 3；
    ratio=1.0 因负贡献无法抵消返回 `no_qualifying_interval`；隐藏墙切分后无合格段、撤销后恢复；
    全负特征返回 `no_positive_contribution`。

## 未完成项 / 限制

1. **Docker 未实测**：本环境无可用 Docker 守护进程（且 8000 端口被其他会话的残留进程占用、
   无权清理），`Dockerfile` / `docker-compose.yml` 未实际构建运行（已做 YAML 语法校验；
   镜像为标准 `python:3.11-slim` + `requirements.txt`，新增代码仅用标准库，无新依赖，风险低）。
   等价地已用 `uvicorn app.main:app` 单 worker 做了完整 HTTP 端到端验证。
2. **单进程内存态**：数据存于进程内存，重启即失；多 worker/多副本不共享状态（compose 固定单容器单 worker）。
   生产化需外置存储或分片路由。
3. **导入耗时**：100,000 镜头导入约 5～7 s（O(N) 建树 + JSON 校验），导入期间会占用 CPU。
4. **定位耗时的常数因子**：为保证严重抵消下的精确性采用 double-double 算术，纯 Python 下
   10 万镜头约 0.8～1 s（已严格 O(N+K)）；若需更低延迟可用 NumPy/Cython 重写内循环，算法不变。
5. 重复导入同一 `video_id` 会**清空其全部订单**（按"整体替换"语义实现，如需保留可改）。
6. 无鉴权、无持久化、无速率限制；`order_id` 仅在单个视频内唯一。
7. 定位对恰好处于阈值（窗口和与 target 仅差几 ulp）的情形采用相对 `positive_total` 的 1e-12
   容差；这是吸收 double 舍入的工程取舍，理论上可能把真实差 1e-12 相对量的临界窗口判为达标。
