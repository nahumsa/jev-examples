# ABCD reference test oracle

Unmodified `utils/process.py` and `utils/evaluate.py` from
https://github.com/asappresearch/abcd/tree/6b8700ce67c6b37b062dd7a60abc76d7ef832a97
are retained as text fixtures under the accompanying MIT license.

Tests compile only the relevant definitions with `ast`, avoiding the upstream
training stack (PyTorch, pandas, sklearn). Feature conversion is replaced with a
recording function; label preparation, value-copy construction, example expansion,
and metrics are executed unchanged. No downloads or paid requests occur in tests.
