# The Drying Floor at Karongi

| Field | Value |
|---|---|
| Country | Rwanda |
| Difficulty | Easy |
| Category | agents / LangGraph state machine debugging |
| Solved by | Nirca |
| Status | solved |
| Flag | `SIGMOID_CONVERGENCE` |

## 60-second version

The LangGraph loop `inspect_batch → dry_batch → inspect_batch` never ended.
`inspect_batch` computed `ready = MOISTURE_MIN <= moisture <= MOISTURE_MAX` but
returned only `{"inspections": ...}`. LangGraph nodes change state **only through
the dict they return**, so `approved` stayed `False` forever, and
`route_after_inspection` always sent the lot back to drying until the recursion
limit (18) was hit. The fix is one line: also return `"approved": ready`.

## The fix

```diff
 def inspect_batch(state: BatchState) -> dict:
     ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX
-    return {"inspections": state["inspections"] + 1}
+    return {"inspections": state["inspections"] + 1, "approved": ready}
```

Nothing else changed, so the structural check against `graph_reference.py` passes.

## Result (`run_log.txt`)

```text
PASS  nyamasheke: ok (passes=0, moisture=12.0, grade=A)
PASS  rutsiro: ok (passes=0, moisture=11.0, grade=A)
PASS  huye: ok (passes=1, moisture=12.5, grade=B)
PASS  nyabihu: ok (passes=2, moisture=12.3, grade=B)
PASS  muhanga: ok (passes=3, moisture=12.5, grade=B)
PASS  karongi: ok (passes=4, moisture=11.8, grade=A)
PASS  gicumbi: ok (passes=5, moisture=12.5, grade=B)
PASS  ngororero: ok (passes=6, moisture=12.5, grade=B)

All 8 batches converged inside the step budget. Flag: SIGMOID_CONVERGENCE
```

## Run

```bash
pip install langgraph==1.2.16 langchain-core==1.6.12
python evaluation_script.py | tee run_log.txt
```

## Files
- `graph.py`: the fixed graph (only `inspect_batch` changed).
- `run_log.txt`: evaluator output, 8/8 PASS and the flag.
- `graph_reference.py`, `evaluation_script.py`, `dataset/batches.json`: supplied, unchanged.
