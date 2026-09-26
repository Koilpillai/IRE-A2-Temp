.PHONY: data retrieve evaluate rerank anti_gaming serving_scale submit test all clean

# One-command rebuild, per Q1.5. Each target is also runnable standalone with
# `--dataset mind` or `--dataset ebnerd` (see README.md) -- `all` runs both
# sequentially, deliberately not in parallel, to stay inside this machine's RAM budget.
#
# Ordering note: `evaluate` (A2 Q4/Q5) reads rerank_val_predictions.parquet and
# behavioral_features_val.parquet, both produced by `rerank` -- so `rerank` MUST run
# before `evaluate` in `all` below, or evaluate's GBDT section silently skips itself
# (it degrades gracefully and prints a warning rather than failing, but the report
# would then be missing the full-two-stage-pipeline numbers Q5 actually asks for).

data:
	python3 -m pipeline.build_pipeline --dataset all

retrieve:
	python3 -m pipeline.retrieval_eval --dataset mind
	python3 -m pipeline.retrieval_eval --dataset ebnerd

rerank:
	python3 -m pipeline.features --dataset all
	python3 -m pipeline.rerank --dataset all
	python3 -m pipeline.ablation --dataset all

evaluate:
	python3 -m pipeline.evaluate --dataset all

anti_gaming:
	python3 -m pipeline.anti_gaming --dataset all

serving_scale:
	python3 -m pipeline.serving_scale --dataset all

submit:
	python3 -m pipeline.generate_submission --dataset mind
	python3 -m pipeline.generate_submission --dataset ebnerd

test:
	python3 -m pytest pipeline/tests/ -v

all: data retrieve rerank evaluate anti_gaming serving_scale test submit

clean:
	rm -rf feature_store/mind/* feature_store/ebnerd/*
	rm -rf outputs/mind/* outputs/ebnerd/*
