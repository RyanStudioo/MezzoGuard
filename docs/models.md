---
layout: page
title: Models
---

Mezzo Guard works with two families of models — **prompt guards** and
**content guards** — and always exposes them through the same unified `Guard`
class. This page lists the built-in models, explains how category
configuration is resolved, and shows how to use your own model.

## Two kinds of guards

- **Prompt guards** protect against harmful prompts: prompt injection,
  jailbreaks, and instructions that try to misuse your application. They score
  one category: `unsafe`.
- **Content guards** block and filter harmful content — quick, easy
  moderation for user-generated text. They score several categories.

It is recommended to use the latest version of each model even if older
versions are supported.

## Supported models

### Prompt guards

Every prompt guard scores a single `unsafe` category.

| Model | Scores |
|-------|--------|
| `RyanStudio/Mezzo-Prompt-Guard-v2-Large` | `unsafe` |
| `RyanStudio/Mezzo-Prompt-Guard-v2-Base` | `unsafe` |
| `RyanStudio/Mezzo-Prompt-Guard-v2-Small` | `unsafe` |
| `meta-llama/Llama-Prompt-Guard-2-22M` | `unsafe` |
| `meta-llama/Llama-Prompt-Guard-2-86M` | `unsafe` |

Collection: [Mezzo Prompt Guard v2 on Hugging Face](https://huggingface.co/collections/RyanStudio/mezzo-prompt-guard-v2).

### Content guards

| Model | Scores |
|-------|--------|
| `RyanStudio/Mezzo-Content-Guard-Large-Preview` | `divisive`, `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-Large` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-Base` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-Small` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-v1.5-Large` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-v1.5-Base` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-v1.5-Small` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-v1.5-Tiny` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-v1.5-Nano` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |
| `RyanStudio/Mezzo-Content-Guard-v1.5-Nano-Preview` | `hate-speech`, `self-harm`, `sexual`, `toxic`, `violence` |

Collections: [Mezzo Content Guard](https://huggingface.co/collections/RyanStudio/mezzo-content-guard)
and [Mezzo Content Guard v1.5](https://huggingface.co/collections/RyanStudio/mezzo-content-guard-v15)
on Hugging Face.

## How categories are resolved

When you construct `Guard(name)`, the category configuration is found in this
order:

1. **An explicit `config` argument** — always wins (see below).
2. **A built-in preset** — the table above is hard-coded in
   `mezzoguard.guard.config.MODELS_CONFIG`, the single source of truth for
   supported ids.
3. **A `.mezzoguard` JSON file** shipped with the model — downloaded from the
   Hugging Face repo (or read from a local model directory).
4. **Nothing found** — a warning is emitted and `guard.config` is `None`:
   scores come back empty, so nothing is ever flagged.

```python
guard = Guard("RyanStudio/Mezzo-Content-Guard-Small")
print(guard.categories)   # ["hate-speech", "self-harm", "sexual", "toxic", "violence"]
```

Model configs (and README deprecation checks) are fetched once and cached per
model name, so repeated construction is cheap after the first call.

## Using a custom model

Three ways, in descending convenience:

### 1. Pass a `GuardConfig`

```python
from mezzoguard import Guard, GuardConfig

guard = Guard(
    "org/my-moderation-model",
    config=GuardConfig(categories=["insult", "threat", "sexual"]),
)
```

A plain dict works too:

```python
guard = Guard("org/my-moderation-model", config={"categories": ["insult", "threat", "sexual"]})
```

### 2. Ship a `.mezzoguard` file with your model

Put a `.mezzoguard` JSON file next to the model weights (in the Hugging Face
repo or a local directory):

```json
{
  "categories": ["insult", "threat", "sexual"],
  "default_max_seq_length": 64,
  "default_overlap": 16,
  "default_replace": "[REDACTED]"
}
```

Only `categories` is required. The other keys are part of the schema;
the SDK methods use their own defaults (64 / 16 / `[REDACTED]`) unless you
pass values explicitly at call time.

A `.mezzoguard` file that is unreadable, not valid JSON, or missing its
`categories` list is ignored with a warning, and the resolution continues as
if the file did not exist (built-in preset next, then the empty-config
fallback of step 4).

### 3. Both

An explicit `config` always wins over a preset or a `.mezzoguard` file.

## Label normalization

Model labels rarely match category names exactly, so labels are normalized
before matching: surrounding whitespace is stripped, everything is
lower-cased, and underscores/spaces become dashes.

| Model label | Matches category |
|-------------|------------------|
| `Hate Speech` | `hate-speech` |
| `self_harm` | `self-harm` |
| `SELF-HARM` | `self-harm` |

Two rules to remember:

- A category named `safe` is **dropped** — safety is the absence of violations,
  not a score. Prompt models that emit a `safe` label simply never report it.
- Labels that are not configured are **ignored** during scoring.

## Deprecation notices

If a model's Hugging Face README declares a replacement in its YAML
frontmatter, constructing a guard for it emits a standard
`DeprecationWarning`:

```
new_version: "RyanStudio/Mezzo-Content-Guard-v1.5-Small"
```

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    guard = Guard("RyanStudio/Some-Old-Guard")
print(caught[0].message)   # This model is deprecated. Use ... instead.
```

## See also

- [Scanning]({{ '/scanning/' | relative_url }}) — using these models.
- [Guard API]({{ '/api/guard/' | relative_url }}) — constructor and `GuardConfig` reference.
