.PHONY: data retrieve evaluate submit test all clean

# One-command rebuild, per Q1.5. Each target is also runnable standalone with
# `--dataset mind` or `--dataset ebnerd` (see README.md) -- `all` runs both
# sequentially, deliberately not in parallel, to stay inside this machine's RAM budget.

data:
	python3 -m pipeline.build_pipeline --dataset all

retrieve:
	python3 -m pipeline.retrieval_eval --dataset mind
	python3 -m pipeline.retrieval_eval --dataset ebnerd

evaluate:
	python3 -m pipeline.evaluate --dataset all

submit:
	python3 -m pipeline.generate_submission --dataset mind
	python3 -m pipeline.generate_submission --dataset ebnerd

test:
	python3 -m pytest pipeline/tests/ -v

all: data retrieve evaluate test submit

clean:
	rm -rf feature_store/mind/* feature_store/ebnerd/*
	rm -rf outputs/mind/* outputs/ebnerd/*
