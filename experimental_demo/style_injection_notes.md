# Style Injection 实验说明

## 1. 实验目标

本实验在现有 `SDXL + ControlNet + IP-Adapter` 风格迁移框架上，加入一种训练无关的 self-attention K/V 风格注入方法，用于探索在不重新训练模型的情况下，是否可以通过局部修改 UNet 中间 attention 特征来增强风格迁移效果。

当前实验配置为：

- 内容图：`contrast_test/dataset/ms_coco_5000`
- 风格图：`contrast_test/dataset/wikiart_5000`
- ControlNet：Canny ControlNet for SDXL
- 基础模型：Stable Diffusion XL base
- IP-Adapter：SDXL IP-Adapter
- AdaIN：开启
- Style Injection：开启
- 注入位置：`up_blocks.0.attentions.1` 中前 5 个 self-attention processor
- 注入强度：`gamma = 0.6`
- 实验名：`proc05_adaintrue_gamma0.6`

输出目录：

```text
experimental_demo/outputs/proc05_adaintrue_gamma0.6_5000/
```

---

## 2. 创新点

### 2.1 从 IP-Adapter/AdaIN 外部再引入 self-attention K/V 注入

原始 InstantStyle / IP-Adapter 主要通过图像编码器提取 style image 的图像特征，再将这些特征作为额外 token 注入 cross-attention 中。AdaIN 版本则进一步在 IP-Adapter 分支中对 hidden states 做均值和方差层面的风格调制。

本实验额外引入了一个新的风格控制入口：

```text
UNet self-attention 的 Key / Value
```

即在生成内容图时，不只依赖 IP-Adapter 的 image prompt token，而是在 UNet 的 self-attention 中直接注入 style image 对应的 attention K/V 特征。

这样做的直觉是：

- self-attention 主要负责 latent 内部空间位置之间的信息聚合；
- style image 的 self-attention K/V 中包含纹理、色彩、局部模式等风格相关信息；
- 在内容图生成过程中，用 style image 的 K/V 影响内容 latent 的 self-attention，可以在不破坏 ControlNet 结构约束的前提下增强风格表达。

---

### 2.2 只在 SDXL 的局部 self-attention 层做注入

与一些全局、多层注入方法不同，本实验不是对所有 self-attention 层进行替换，而是限制在：

```text
up_blocks.0.attentions.1
```

并进一步精确到其中若干个 transformer block 的 `attn1.processor`。

当前 5 processor 配置为：

```text
up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor
up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor
up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor
up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor
up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor
```

其中：

- `attn1` 表示 self-attention；
- `attn2` 表示 cross-attention；
- 本实验只操作 `attn1`，不直接修改文本 cross-attention。

这种局部注入的优点是：

1. 对原 SDXL 生成过程破坏更小；
2. 更容易控制变量；
3. 便于做 1 / 5 / 10 个 processor 的消融实验；
4. 可以观察不同注入层数对风格强度和结构保持的影响。

---

### 2.3 每个 denoising step 都进行 style forward，动态缓存当前 timestep 的 K/V

本实验不是只提取一次 style K/V 后重复使用，而是在每个 denoising timestep 都执行一次 style forward。

流程为：

```text
当前 timestep t
    ↓
style image -> VAE latent z_s
    ↓
根据当前 sigma 加噪得到 z_s_t
    ↓
style forward through UNet
    ↓
缓存目标 self-attention processor 的 K_s^t, V_s^t
    ↓
content forward through UNet
    ↓
在同一 timestep 注入 K_s^t, V_s^t
```

这样可以保证 style K/V 和当前生成 timestep 对齐，比固定 timestep 的 K/V 缓存更接近 diffusion 过程。

代价是每一步多一次 style UNet forward，因此推理速度大约会变慢。

---

### 2.4 支持 K/V 同时注入、只注入 V、以及多 processor 消融

实现中支持以下配置：

```yaml
style_injection:
  enabled: true
  components: key_value
  gamma: 0.6
```

其中 `components` 可以为：

```text
value      # 只注入 V，更稳定
key_value  # 同时注入 K 和 V，风格影响更强
```

当前 5000 张批量实验使用的是：

```yaml
components: key_value
gamma: 0.6
```

即同时混合 style K 和 style V。

混合公式为：

```text
K_new = (1 - gamma) * K_content + gamma * K_style
V_new = (1 - gamma) * V_content + gamma * V_style
```

当 `components: value` 时，只修改 V：

```text
K_new = K_content
V_new = (1 - gamma) * V_content + gamma * V_style
```

---

## 3. 具体实现方法

### 3.1 自定义 StyleKVInjectionController

在 `experimental_demo/infer_style_controlnet.py` 中新增了 `StyleKVInjectionController`，用于管理：

- 当前阶段：`style` 或 `content`
- 当前 timestep
- 每个 timestep 缓存的 style K/V
- 注入强度 `gamma`
- 注入组件 `components`
- 调试统计：`capture_calls`、`inject_calls`、`avg_key_delta`、`avg_value_delta`

核心逻辑：

```text
style phase:
    缓存当前 timestep 的 K_style, V_style

content phase:
    读取同一 timestep 的 K_style, V_style
    按 gamma 混合到 content K/V
```

---

### 3.2 自定义 StyleKVSelfAttnProcessor

为了只修改目标 self-attention 层，实验中定义了一个新的 attention processor：

```text
StyleKVSelfAttnProcessor
```

它替换 SDXL UNet 中指定的 `attn1.processor`。

在 forward 中：

1. 计算内容或风格 latent 的 Query / Key / Value；
2. 如果当前是 style phase，则缓存 K/V；
3. 如果当前是 content phase，则取出 style K/V 并混合；
4. 使用修改后的 K/V 执行 scaled dot-product attention。

伪代码如下：

```python
query = to_q(hidden_states)
key = to_k(hidden_states)
value = to_v(hidden_states)

if phase == "style":
    cache[timestep] = {"key": key, "value": value}

if phase == "content":
    style_key, style_value = cache[timestep]
    key = (1 - gamma) * key + gamma * style_key
    value = (1 - gamma) * value + gamma * style_value

hidden_states = attention(query, key, value)
```

---

### 3.3 对 UNet forward 做轻量 monkey patch

为了在每个 denoising step 之前先做 style forward，实验脚本临时包裹了 `pipe.unet.forward`。

新的流程是：

```text
style_injection_unet_forward(sample, timestep, ...):
    1. 用 style latent 构造当前 timestep 的 noisy style latent
    2. phase = style
    3. original_unet_forward(style_noisy_latents, timestep, ...)
    4. phase = content
    5. original_unet_forward(content_sample, timestep, ...)
```

这样不需要改 diffusers 源码，也不会影响根目录的 `infer_style_controlnet.py` 或 `gradio_demo`。

---

### 3.4 style latent 的构造方式

对每张 style image：

1. 读取 style image；
2. resize 到当前输出尺寸；
3. 归一化到 `[-1, 1]`；
4. 通过 SDXL VAE encoder 得到 style latent：

```text
z_s = VAE.encode(style_image)
```

5. 在每个 timestep 根据 Euler scheduler 的 sigma 构造 noisy style latent：

```text
z_s_t = (z_s + sigma_t * noise) / sqrt(sigma_t^2 + 1)
```

这样得到和当前 denoising step 尺度更接近的 style latent。

---

### 3.5 JSONL 批处理

为了跑 5000 张，创建了专用 JSONL：

```text
experimental_demo/proc05_adaintrue_gamma0.6_pairs_5000.jsonl
```

每行包含：

```json
{
  "style_image_path": "wikiart_5000/xxxxx.jpg",
  "control_image_path": "ms_coco_5000/xxxxx.jpg",
  "output_image_path": "experimental_demo/outputs/proc05_adaintrue_gamma0.6_5000/xxxxx.jpg",
  "prompt": "COCO caption",
  "adain_ip": true,
  "style_injection_gamma": 0.6,
  "style_injection_components": "key_value",
  "style_injection_match_stats": false,
  "style_injection_processor_names": [
    "up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor"
  ]
}
```

专用配置文件为：

```text
experimental_demo/proc05_adaintrue_gamma0.6_config.yaml
```

专用推理入口为：

```text
experimental_demo/infer_proc05_adaintrue_gamma0.6_5000.py
```

运行命令：

```bash
cd /root/autodl-tmp/InstantStyle-main
python experimental_demo/infer_proc05_adaintrue_gamma0.6_5000.py
```

---

## 4. 与原始代码的关系

本实验只在 `experimental_demo` 下新增和修改实验代码，不影响：

```text
gradio_demo/
infer_style_controlnet.py
ip_adapter/
```

主要实验文件：

```text
experimental_demo/infer_style_controlnet.py
experimental_demo/proc05_adaintrue_gamma0.6_config.yaml
experimental_demo/proc05_adaintrue_gamma0.6_pairs_5000.jsonl
experimental_demo/infer_proc05_adaintrue_gamma0.6_5000.py
```

---

## 5. 当前实验配置总结

当前 5000 张实验 `proc05_adaintrue_gamma0.6` 的关键设置：

```yaml
ip_adapter:
  adain_ip: true
  adain_alpha: 1.0
  adain_beta: 1.0

style_injection:
  enabled: true
  components: key_value
  gamma: 0.6
  match_stats: false
  processor_names:
    - up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor
    - up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor
    - up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor
    - up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor
    - up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor
```

含义：

- 使用 IP-Adapter 提供全局风格条件；
- 使用 AdaIN 增强 IP-Adapter 分支的风格统计迁移；
- 额外在 SDXL UNet 的 5 个 self-attention processor 中注入 style K/V；
- K 和 V 同时注入；
- style K/V 占比为 60%。

---

## 6. 后续可做的消融实验

建议后续比较以下设置：

1. 无 style injection，仅 IP-Adapter；
2. IP-Adapter + AdaIN；
3. IP-Adapter + style injection，只注入 V；
4. IP-Adapter + style injection，同时注入 K/V；
5. processor 数量：1 / 5 / 10；
6. gamma：0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0；
7. `match_stats: true` 与 `false` 的对比。

这些实验可以用于判断：

- K/V 注入是否增强风格；
- 是否破坏内容结构；
- AdaIN 与 self-attention K/V injection 是否互补；
- 不同 processor 数量对结果的影响。
