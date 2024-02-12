#!/bin/bash

python -m pyserini.search.faiss \
  --index wikipedia-dpr-100w.dpr-multi \
  --topics dpr-nq-test \
  --encoded-queries dpr_multi-nq-test \
  --output runs/run.dpr.nq-test.multi.bf.trec \
  --batch-size 36 --threads 12