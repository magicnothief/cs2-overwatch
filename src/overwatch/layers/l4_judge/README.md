# Layer 4: LLM judge

**Purpose:** turn a Moment's evidence into readable reasoning plus a verdict label, as JSON.
The calibrated **probability** comes from Layer 3 (see `docs/ARCHITECTURE.md` §3.2), unless you
measure that the LLM's token probability adds signal.

**Learn first:** prompt design; structured output (JSON schema / llama.cpp grammars);
GGUF and quantization; LoRA/QLoRA; distillation; LLM calibration (token logprobs).

**Exercises, in order:**
1. Write `render_evidence(moment) -> str` and read 20 outputs yourself. If *you* couldn't judge
   from that text, the model can't either. Fix the renderer first.
2. **Zero-shot baseline:** an off-the-shelf ~4B GGUF in llama.cpp with your prompt. Measure it
   on the golden set.
3. Only then fine-tune (`training/llm/`) and compare against step 2.

**Done when:**
- It beats the zero-shot baseline.
- The output is valid JSON 100% of the time.
- The reasoning never cites a fact that isn't in the evidence (check this by hand on a sample).

**Pitfalls:**
- Teacher rationales that "know" the label.
- The model learning "high HS% = cheater".
- CPU latency. Judge only the top few moments per player.
