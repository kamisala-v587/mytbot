# BPVA 数据测评

推荐直接显式运行 Python 命令，这样参数都能一眼看见，也更方便临时改动。下面的命令与当前 benchmark 默认值一致。

## 快速开始

CPU 单进程示例：

```bash
cd /vla/workspace/my_tbot
conda activate mytbot
python -m tools.bpva_benchmark.data_benchmark --config-path configs/Pro6k/bench_compare/bpva_bench.jsonc  --warmup-batches 10   --measure-batches 100
```
