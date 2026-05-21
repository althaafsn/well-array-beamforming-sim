from __future__ import annotations

import ast
from pathlib import Path


def test_imaging_module_does_not_import_wall_oracle() -> None:
    source = Path("src/well_array_sim/internal/imaging.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "inner_radius_at" not in names
    assert "WallProfile" not in names
    assert "ground_truth" not in names
