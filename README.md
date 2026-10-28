# idocr — 身份证识别（RapidOCR + 锚点行 + MOD 11-2 多预处理投票）

纯 CPU 跑的身份证字段抽取：输入一张身份证照片（人像面/国徽面、彩色/黑白、
模糊/反光/倾斜/翻拍都行），输出 8 个结构化字段，并追加写入 CSV。

没有 GPU、CUDA、模型移植，也没有任何需要"改模型"的步骤——整条链路是
**现成 OCR 引擎 + 传统 CV + 规则**。

```
pip install -r requirements.txt
python run.py 你的图片目录 -o results.csv
```

---

## 为什么是「抽取式」而不是「端到端 VLM」

生成式模型在身份证号上最危险的失败方式是**静默编造**：给你一个格式完全合法、
但某几位是猜的号码，还不会告诉你哪几位不确定。这个项目反过来做：

| 环节 | 做法 | 得到什么 |
|---|---|---|
| 检测 | 找最大四边形（长宽比 1.585 先验） | 透视矫正，不依赖模型 |
| 判断正反面 | 看 OCR 文本里有"签发机关/有效期限"还是"姓名/公民身份号码" | 省掉一个分类模型 |
| 定位字段 | 锚点行（"姓名""公民身份号码"…）+ 正则 | 换版式也能用，不用像素模板 |
| 识别 | RapidOCR（ONNXRuntime，CPU） | 离线、无 GPU |
| 提准 | 4 种预处理变体分别识别，按位加权投票 | 用"多视角"消解单次随机错误 |
| 判定 | MOD 11-2 校验位 + 出生/性别/行政区划交叉验证 | 错了能被发现 |
| 兜底 | 校验不过 → `needs_review=1` | 交人工，绝不假装成功 |

关键性质：**它宁可说"我不确定"，也不会给你一个看起来对的错号码。**

---

## 目录结构

```
idocr/
├── run.py                  # 入口：python run.py <图片或目录> -o results.csv
├── requirements.txt
├── results.example.csv     # 示例输出（7 张合成样本）
├── idocr/
│   ├── ocr.py              # RapidOCR 适配（兼容 rapidocr / rapidocr_onnxruntime）
│   ├── preprocess.py       # 图像读写 + 预处理变体
│   ├── card.py             # 四边形检测、透视矫正、180° 方向校正
│   ├── extract.py          # 行聚类、锚点解析、各字段正则
│   ├── validate.py         # MOD 11-2、受约束纠错、交叉校验
│   ├── pipeline.py         # 编排 + 加权投票
│   ├── csvout.py           # CSV 追加写（UTF-8 BOM，Excel 直接打开）
│   └── cli.py
└── tests/
    ├── make_sample.py      # 生成合成样本（清晰/旋转/模糊/反光/摆拍透视）
    ├── test_logic.py       # 16 个纯逻辑用例，不需要 OCR
    └── test_e2e.py         # 3 个端到端用例，需要 OCR（约 15 秒）
```

---

## 安装

```bash
pip install -r requirements.txt
```

`rapidocr_onnxruntime` 自带检测/识别模型，装完即可离线用（约 15 MB）。
若想用新版 `rapidocr`，代码会自动优先使用它。

> Windows 上中文路径没问题：读写走 `np.fromfile + cv2.imdecode`，没有用
> `cv2.imread`。

## 快速开始

```bash
# 1) 生成一组合成样本（没有真实样本时先跑通流程）
python tests/make_sample.py

# 2) 识别并写入 CSV
python run.py tests/samples -o results.csv

# 3) 想看某张图为什么失败
python run.py tests/samples/front_blur.png -o results.csv --debug-dir debug
#   debug/ 下会有：矫正后的卡片、每个预处理变体、以及 OCR 全文
```

输出示例（节选）：

```
[1/7]   OK   back_clear.png     id=-                    name=-      (4.15s)
[4/7]   OK   front_clear.png    id=110105199003071239   name=张伟   (2.98s)
[6/7]   OK   front_on_desk.png  id=110105199003071239   name=张伟   (3.05s)

完成：7 张，校验通过 5，待复核 0
```

---

## 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `paths` | 必填 | 图片文件或目录，可多个，目录会递归 |
| `-o/--csv` | `results.csv` | 结果文件，**追加写**，不存在时自动建表头 |
| `--variants` | `orig,clahe,sharpen,binarize` | 预处理变体，`orig` 必留。可选 `gray,upscale,denoise` |
| `--debug-dir` | 无 | 保存矫正图/变体图/OCR 全文，调参用 |
| `--no-detect` | 关 | 跳过四边形检测，直接缩放（图已经是裁剪好的卡片时更快） |
| `--min-confidence` | 0.5 | 字段置信度低于此值 → 标记待复核 |
| `--jsonl` | 无 | 同时输出 JSONL 明细 |
| `--quiet` | 关 | 只打印汇总 |

变体越多越准、越慢。**建议**：模糊/低质多就用默认四个；追求速度用
`--variants orig,clahe`（约快一倍）。

---

## CSV 字段

| 列 | 含义 |
|---|---|
| `timestamp` | 处理时间 |
| `image` | 图片路径 |
| `side` | `front` / `back` / `both` / `unknown` |
| `mode` | 预处理方式：`warp`(检测到卡片并矫正) / `resize`(已是卡片比例) / `raw`(原图) |
| `name` `gender` `ethnicity` `birth_date` `address` `id_number` | 人像面字段，`birth_date` 统一 `YYYY-MM-DD` |
| `issuing_authority` `valid_period` | 国徽面字段，`valid_period` 统一 `YYYY.MM.DD-YYYY.MM.DD` 或 `...-长期` |
| `id_checksum_ok` | 1 = 通过 MOD 11-2 |
| `id_repaired` | 1 = 结果经过（受约束的）修正 |
| `conf_name` `conf_id_number` | 变体间一致性（0–1） |
| `needs_review` | **1 = 必须人工复核**，不要直接入库 |
| `notes` | 诊断信息，如 `birth_mismatch`、`id_unrecoverable`、`low_confidence:address` |
| `elapsed_s` | 单张耗时 |

入库时请以 `needs_review == 0 and id_checksum_ok == 1` 作为自动通过的条件，
其余全部进人工队列。

---

## 关于校验位：能做什么，不能做什么

这是本项目最容易被误解决的地方，代码里也写了注释。

**能做**：MOD 11-2 的权重向量每个元素都与 11 互素，所以**任意单字符错误都会
改变校验值**——校验失败是可靠的错误信号。

**不能做**：不要指望"逐位替换、直到校验通过"来纠错。第 i 位换成某个数字后
校验通过的概率约 1/11，17 个位置扫一遍几乎必然出现十几个"看似合法"的解，
无法判断哪个是真的——盲目搜索只会把错号码变成**合法但错误**的号码，比原来
更危险。

所以 `validate.repair()` 只在一种情况下动手：**多个预处理变体在个别位上出现
分歧**，用这些分歧字符做替换，且必须**恰好一个解**才采信；否则一律 `None`，
交给人工。这也是 `id_repaired` 很少为 1 的原因——正常，说明系统偏保守。

---

## 性能参考

本机（普通 x86 CPU，4 个变体，单进程）实测 **2.5–4.1 秒/张**，含卡片检测与
4 次 OCR。按这个量级：

- 单进程约 900–1400 张/小时；
- 4 进程（`--variants orig,clahe`）轻松 5000+ 张/小时；
- 多进程直接开多个进程跑不同目录即可，CSV 追加写在同一路径也能工作（行级
  追加，不建议超高并发写同一个文件）。

证件是低频场景，这个吞吐通常过剩。

---

## 调优建议

1. **先量基线**：拿你自己 200 张真实图跑一遍，统计 `needs_review` 比例和人工
   复核后的错误率。不要看任何公开 benchmark。
2. **失败的图先看 `--debug-dir`**：
   - `*__orig.png` 是矫正后的卡片。如果它是歪的/裁错了 → 问题在 `card.py`；
   - `*__text.txt` 是 OCR 全文。如果锚点读错（"住址"→"住扯"）→ 在
     `extract.py` 的锚点/正则里加容错；
   - 如果文本对但字段没抽出来 → 问题在解析规则。
3. **反光严重**：加 `--variants orig,clahe,binarize,denoise`，denoise 对摩尔纹
   有效但慢。
4. **超低分辨率**（宽度 < 600）：`upscale` 变体几乎是必需的。
5. **阈值**：`--min-confidence` 调高会更保守（更多 review），调低则相反。

---

## 已知限制（请按需评估）

- **90° 横置的卡片不做旋转矫正**：检测依赖横向长宽比，竖着拍的卡会退回
  `resize`；只处理 180° 颠倒。竖拍场景请在上游先转正。
- **一图多卡（混贴）不支持**：只取最可信的一张。
- **15 位老身份证**：能输出，但 MOD 11-2 不适用，`id_checksum_ok` 会是 0。
- **号码被遮挡/裁切**：无法纠错，直接 `needs_review`。
- **手写、生僻字姓名**：准确率取决于 OCR 引擎本身，不是本项目的强项。
- **屏幕翻拍摩尔纹**：可能识别失败；失败是安全的（进 review），不会编造。

---

## 部署在平头哥机器上

这套代码**完全不需要 PPU**：它跑在宿主机 x86 CPU 上，与 PPU 上跑的文档解析
（MinerU 等）互不干扰。同一台机器两个容器：

```bash
# 证件链路（CPU）
python run.py /data/idcards -o /data/idcards/results.csv

# 文档链路（PPU，另见 MinerU 的 ppu.Dockerfile）
```

好处是合规：证件不出机器、不进 GPU 驱动栈，审计面小。

---

## 合规提醒

身份证号是敏感个人信息。本项目默认**只写 CSV，不保存原图**；`--debug-dir`
会把卡片图和 OCR 全文落盘，生产环境请关掉，或落在加密盘并设保留期。
日志与 CSV 建议对号码做脱敏（保留前 6 后 4）。

---

## 测试

```bash
python tests/test_logic.py   # 16 个用例，0.1 秒，不需要 OCR
python tests/test_e2e.py     # 3 个用例，约 15 秒，需要 OCR
```

`test_e2e.py` 覆盖：清晰人像面、清晰国徽面、模糊/旋转/摆拍透视三种退化。
