# LLM 言い換え生成用的固定提示词

用途：为 `prompts.yaml` 中每个类别的 `llm` 字段生成 12 条言い換え。
**五个类别使用完全相同的提示词，只替换 `{...}` 部分，且各开一个新对话。**
不要追加额外指示、不要挑选"看起来好的"输出、不要手动润色，原样粘贴前 12 条。
这样做是为了保证"LLM 条件"是一个可复现的自动过程，而不是第二轮人工写作。

在 `prompts.yaml` 的 `meta.llm_for_paraphrase` 里记录模型名称和日期。

---

## 提示词（复制以下全部内容，替换 {…} 后发送）

You are helping to build text prompts for a vision-language model (CLIP) that must recognize a type of damage on concrete surfaces from photos.

Below are visual descriptions of one damage type, written by an engineer from a Japanese bridge inspection manual. The damage name is deliberately NOT given.

Descriptions:
{把该类别 attr 字段中的 10 条 text 逐行粘贴在这里}

Task: write 12 new English descriptions of what this damage LOOKS LIKE in a photo.

Requirements:
1. Each description must be a noun phrase that reads naturally after both "a concrete surface with ..." and "..., which has ...". Do not start with "a photo of".
2. Describe only visible properties: shape, width, direction, color, texture, distribution, and where it appears. No causes, no materials science, no severity judgments.
3. Do NOT use the name of the damage or its common synonyms: {该类别的禁用词，例如 crack 类填: crack, cracking, fissure, fracture}.
4. Use everyday words that commonly appear in web image captions; avoid technical engineering terms.
5. Vary the level of detail: 4 short descriptions (3–6 words), 4 medium (7–12 words), 4 detailed (13–20 words).
6. Do not repeat the input descriptions verbatim.

Output format: a JSON list of 12 strings and nothing else.

---

## 禁用词参考（第 3 条）

| 类别 | 禁用词 |
|---|---|
| crack | crack, cracking, cracked, fissure, fracture |
| spallation | spalling, spall, spallation, delamination, flaking |
| efflorescence | efflorescence, free lime, calcium, salt deposit |
| exposed_bars | exposed rebar, rebar, reinforcement, reinforcing bar |
| corrosion_stain | corrosion, rust stain, rust |

注：禁用词是为了让 `llm` 条件与 `attr` 条件一样"不含类别名"，从而检验"只靠描述能否把概念传给模型"。
含类别名的版本由代码自动构造（`label_llm` 条件）。
