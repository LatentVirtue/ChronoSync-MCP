"""Core engine: profiling, alignment, reliability auditing, synthetic benchmarks.

Invariant: no module in this package may import `mcp`. The MCP layer wraps this
engine; the engine must stay usable from plain Python and the CLI.
"""
