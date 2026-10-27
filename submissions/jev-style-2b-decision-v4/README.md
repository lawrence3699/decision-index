# Jev-Style-2B-Decision-v4

[chaoliangUNSW/Jev-Style-2B-Decision-v4 @ `3f2c063f7d94cfa281b9fee87a5b89bed6745506`](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4/tree/3f2c063f7d94cfa281b9fee87a5b89bed6745506),
Apache-2.0. A full fine-tune of Qwen/Qwen3.5-2B with its vision tower (frozen); it returns a probability per option and
generates no text. The repo's current head (`c9504148`) changes only `README.md` and `manifest.json`; the weights and
`src/` are the same.

## Running it

The engines are `macjev.eval.di_v4:V4TextEngine` and `macjev.eval.di_v4:V4VisionEngine`, in the model repo under
`src/`: the code the model was trained, calibrated and evaluated with. Install the model's requirements next to the
kit (`25c6ec7`):

```sh
hf download chaoliangUNSW/Jev-Style-2B-Decision-v4 --revision 3f2c063f7d94cfa281b9fee87a5b89bed6745506 --local-dir jev4
pip install -r jev4/requirements.txt flash-linear-attention==0.6.0
```

The runs used torch 2.14.1+cu130, transformers 5.17.0, flash-linear-attention 0.6.0 (the Gated DeltaNet kernels;
without it transformers falls back to its torch path) and triton 3.8.0, without causal-conv1d.

### Text (public 0.3 suite)

```sh
PYTHONPATH=jev4/src python -m decision_index pipeline --engine macjev.eval.di_v4:V4TextEngine \
  --option model=jev4 --option device=cuda --option precision=bf16 --option fast=true \
  --compact --out runs/jev-style-2b-decision-v4
```

`fast=true`: each request's state is computed once with the training block attention and cached, and the questions'
own blocks run on that cache, batched. The submitted run used this engine and these options with the kit's
`decision_index.runner.run` in 5 processes on one GPU, each over the rows with `sha256(group_id) mod 5 == k` (linked
cases stay together); the five `results.jsonl` were concatenated (every `run_id` once) and scored with the kit.

For the latency measurement add `--option cuda_graphs=true`, the fastest path. At load the engine records one CUDA
graph per padded row length (64 to 8,192 tokens; about a minute on a fresh machine while Triton compiles its kernels,
seconds after that). Single-question requests replay one graph; requests with several questions use the graphs only
while that is cheaper than the shared-state path.

### Vision (public vision suite, rebuilt and verified in `work/vision`)

```sh
PYTHONPATH=jev4/src python -m decision_index vision pipeline --dir work/vision \
  --engine macjev.eval.di_v4:V4VisionEngine --option model=jev4 --option device=cuda --option precision=bf16 \
  --option batch_tokens=32768 --compact --out runs/jev-style-2b-decision-v4-vision
```

Images go in at native resolution capped at 1,638,400 pixels (the engine's default `max_pixels`), processed by the
checkpoint's own image processor. The submitted run is this command followed by the rows not yet answered, split by
`sha256(row id) mod N` and run with the kit's `decision_index.vision.run.run` (same engine and options), one process
each on the same GPU; the parts were merged with every row id once and scored with the kit's `score_run`.
