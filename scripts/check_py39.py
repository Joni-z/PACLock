"""Fail on PEP 604 annotations in modules that cannot evaluate them.

The cluster runs Python 3.9, where `int | None` in a signature is evaluated at
definition time and raises TypeError unless the module imports
`annotations` from __future__. This has now cost two rounds of dead jobs -- once
in head.py (eight candidate jobs died 13 seconds in) and once in
frontend/triaxial.py. ast.parse accepts the syntax, so only an import catches it
at runtime and only this catches it before submission.
"""
import ast
import pathlib
import sys

bad = []
for path in sorted(pathlib.Path("paclock_bench").rglob("*.py")):
    src = path.read_text()
    tree = ast.parse(src)
    has_future = any(
        isinstance(n, ast.ImportFrom) and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in tree.body
    )
    if has_future:
        continue
    for node in ast.walk(tree):
        # `X | Y` inside an annotation context
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            anns = [a.annotation for a in node.args.args + node.args.kwonlyargs
                    if a.annotation] + ([node.returns] if node.returns else [])
            for ann in anns:
                if any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.BitOr)
                       for x in ast.walk(ann)):
                    bad.append(f"{path}:{node.lineno} {node.name}()")
                    break

if bad:
    print("PEP 604 annotation without `from __future__ import annotations`:")
    for b in bad:
        print("  ", b)
    sys.exit(1)
print(f"ok: no py3.9-incompatible annotations")
