# GUI app to test 3 LLMs for data linkage

A GUI-based tool for linking names across two datasets using powerful open-source Large Language Models (LLMs). It leverages reasoning and contextual clues (like Department and School) to match entries, even when names don’t exactly match. Note - testing is underway. 

## Supported Models

- **Phi-2** (Microsoft)
- **Mistral-7B Instruct (Quantized)** - via GPTQ
- **OpenChat 3.5 (Quantized)** - via GPTQ

## Features

- Match names using natural language reasoning, not just exact or fuzzy logic.
- Use additional fields (like department and school) to improve matching accuracy.
- Interactive GUI built with Tkinter.
- Save results with detailed LLM outputs to a CSV file.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/byoules/llm-name-linker.git
cd llm-name-linker
```

### 2. Install Dependencies
```bash
pip install pandas transformers accelerate auto-gptq bitsandbytes tkinter
```

### 3. Getting started

1. Select Dataset 1 (the reference list) - mockdataset1.csv
2. Select Dataset 2 (to be matched). - mockdataset2.csv
3. Choose a model:
- Phi-2
- Mistral-7B (quantized)
- OpenChat 3.5 (quantized)

### 4. Click 'Run Linking'.

The tool will analyze each possible pair of names and generate a result using the LLMs.

We may need to tweak the prompts a bit. 

### 5. Export
The output will be saved as `linked_name_results.csv`
