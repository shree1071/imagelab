"""Tests for Python Pipeline Exporter service and API endpoint."""

import ast
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from app.models.graph import GraphEdge, GraphNode, PipelineGraph
from app.models.pipeline import PipelineStep
from app.operators.registry import OPERATOR_REGISTRY
from app.services.python_exporter import (
    export_graph_to_python,
    export_pipeline_to_python,
    get_unsupported_operators,
)

EXPORT_ENDPOINT = "/api/v1/pipeline/export-python"


def test_all_registered_operators_emit_valid_python_syntax():
    """Verify that every single operator in OPERATOR_REGISTRY generates valid Python AST."""
    for op_name in OPERATOR_REGISTRY:
        step = PipelineStep(type=op_name, params={})
        code, unsupported = export_pipeline_to_python([step], f"Test {op_name}")
        assert unsupported == [], f"Operator {op_name} reported as unsupported"
        try:
            tree = ast.parse(code)
            assert isinstance(tree, ast.Module)
        except SyntaxError as e:
            raise AssertionError(f"Operator {op_name} produced invalid Python code:\n{code}") from e


def test_unsupported_operator_reported_and_commented():
    """Verify unsupported operator is recorded and gracefully commented out."""
    step = PipelineStep(type="non_existent_future_operator", params={"val": 42})
    unsupported_list = get_unsupported_operators([step])
    assert unsupported_list == ["non_existent_future_operator"]

    code, unsupported = export_pipeline_to_python([step], "Unknown Operator Pipeline")
    assert unsupported == ["non_existent_future_operator"]
    assert "Unsupported operator 'non_existent_future_operator'" in code

    # The rest of the script is still syntactically valid
    ast.parse(code)


def test_macro_blend_code_generation():
    """Verify macro_blend generates branch logic, dimension checks, and cv2.addWeighted."""
    step = PipelineStep(
        type="macro_blend",
        params={"alpha": 0.6, "beta": 0.4, "gamma": 1.0},
        branches={
            "left": [PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 5, "heightSize": 5})],
            "right": [PipelineStep(type="imageconvertions_grayimage", params={})],
        },
    )
    code, unsupported = export_pipeline_to_python([step], "Blend Pipeline")
    assert unsupported == []
    assert "cv2.addWeighted" in code
    assert "cv2.resize" in code
    assert "0.6" in code
    assert "0.4" in code
    ast.parse(code)


def test_macro_if_else_code_generation():
    """Verify macro_if_else generates conditional branching."""
    step = PipelineStep(
        type="macro_if_else",
        params={"metric": "mean_brightness", "comparator": ">", "threshold": 128},
        branches={
            "then": [PipelineStep(type="imageconvertions_invertimage", params={})],
            "else": [PipelineStep(type="blurring_applymedianblur", params={"kernelSize": 5})],
        },
    )
    code, unsupported = export_pipeline_to_python([step], "If Else Pipeline")
    assert unsupported == []
    assert "if float(cv2.mean(image)[0]) > 128.0:" in code
    assert "cv2.bitwise_not(image)" in code
    assert "cv2.medianBlur" in code
    ast.parse(code)


def test_numpy_import_deduplication():
    """Only import numpy if operators require numpy features."""
    # Pipeline without numpy requirements
    pure_cv_steps = [
        PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 3, "heightSize": 3}),
        PipelineStep(type="imageconvertions_grayimage", params={}),
    ]
    code_pure, _ = export_pipeline_to_python(pure_cv_steps)
    assert "import cv2" in code_pure
    assert "import numpy as np" not in code_pure

    # Pipeline with numpy requirement (sharpen filter)
    sharpen_steps = [
        PipelineStep(type="filtering_sharpen", params={"strength": 2.0}),
    ]
    code_np, _ = export_pipeline_to_python(sharpen_steps)
    assert "import cv2" in code_np
    assert "import numpy as np" in code_np


def test_export_graph_to_python():
    """Verify PipelineGraph can be compiled and exported directly."""
    graph = PipelineGraph(
        nodes=[
            GraphNode(id="n1", type="blurring_applyblur", params={"widthSize": 3, "heightSize": 3}),
            GraphNode(id="n2", type="imageconvertions_grayimage", params={}),
        ],
        edges=[
            GraphEdge(from_node="n1", to_node="n2"),
        ],
    )
    code, unsupported = export_graph_to_python(graph, pipeline_name="Graph Export Pipeline")
    assert unsupported == []
    assert "cv2.blur" in code
    assert "Convert to Grayscale" in code
    ast.parse(code)


def test_end_to_end_script_execution():
    """Execute the exported Python script in a subprocess and verify output image generation."""
    steps = [
        PipelineStep(
            type="macro_blend",
            params={"alpha": 0.5, "beta": 0.5},
            branches={
                "left": [PipelineStep(type="blurring_applygaussianblur", params={"widthSize": 5, "heightSize": 5})],
                "right": [PipelineStep(type="imageconvertions_grayimage", params={})],
            },
        ),
        PipelineStep(
            type="macro_if_else",
            params={"metric": "mean_brightness", "comparator": ">", "threshold": 50},
            branches={
                "then": [PipelineStep(type="imageconvertions_invertimage", params={})],
                "else": [PipelineStep(type="filtering_sharpen", params={"strength": 1.0})],
            },
        ),
    ]

    code, _ = export_pipeline_to_python(steps, "Execution Test Pipeline")

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "run_script.py")
        input_path = os.path.join(tmpdir, "test_input.png")
        output_path = os.path.join(tmpdir, "test_output.png")

        # Create synthetic 80x80 BGR image
        test_img = np.full((80, 80, 3), 150, dtype=np.uint8)
        cv2.imwrite(input_path, test_img)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, script_path, "--input", input_path, "--output", output_path],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert os.path.exists(output_path), "Output image was not generated"

        saved_img = cv2.imread(output_path)
        assert saved_img is not None
        assert saved_img.shape[:2] == (80, 80)


def test_api_export_python_pipeline_request(client):
    """Test POST /api/v1/pipeline/export-python with pipeline steps."""
    payload = {
        "pipeline_name": "My Custom Pipeline",
        "input_filename": "source.png",
        "output_filename": "dest.png",
        "pipeline": [
            {"type": "blurring_applygaussianblur", "params": {"widthSize": 5, "heightSize": 5}},
            {"type": "imageconvertions_grayimage", "params": {}},
        ],
    }
    res = client.post(EXPORT_ENDPOINT, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["filename"] == "my_custom_pipeline.py"
    assert data["unsupported_operators"] == []
    assert "cv2.GaussianBlur" in data["code"]
    ast.parse(data["code"])


def test_api_export_python_graph_request(client):
    """Test POST /api/v1/pipeline/export-python with graph."""
    payload = {
        "pipeline_name": "Graph Workflow",
        "graph": {
            "nodes": [
                {"id": "node-1", "type": "blurring_applyblur", "params": {"widthSize": 3, "heightSize": 3}},
                {"id": "node-2", "type": "imageconvertions_grayimage", "params": {}},
            ],
            "edges": [{"from_node": "node-1", "to_node": "node-2"}],
        },
    }
    res = client.post(EXPORT_ENDPOINT, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["filename"] == "graph_workflow.py"
    assert "cv2.blur" in data["code"]


def test_api_export_python_missing_payload(client):
    """Test POST /api/v1/pipeline/export-python with neither pipeline nor graph fails validation."""
    payload = {"pipeline_name": "Empty"}
    res = client.post(EXPORT_ENDPOINT, json=payload)
    assert res.status_code == 422


def test_special_characters_and_quotes_in_pipeline_name(client):
    """Verify quotes, parentheses, and existing .py suffix in pipeline name are handled cleanly."""
    payload = {
        "pipeline_name": 'My "Super" Filter (v2.1).py',
        "pipeline": [{"type": "blurring_applyblur", "params": {"widthSize": 3, "heightSize": 3}}],
    }
    res = client.post(EXPORT_ENDPOINT, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "my_super_filter_v2_1.py"
    # Verify no syntax error is caused by quotes inside argparse description
    ast.parse(data["code"])


def test_comparator_normalization_in_macro_if_else():
    """Verify '=' is normalized to '==' and invalid comparators fall back safely."""
    step_equal = PipelineStep(
        type="macro_if_else",
        params={"metric": "mean_brightness", "comparator": "=", "threshold": 100},
        branches={"then": [], "else": []},
    )
    code_eq, _ = export_pipeline_to_python([step_equal])
    assert "== 100.0:" in code_eq
    ast.parse(code_eq)

    step_fallback = PipelineStep(
        type="macro_if_else",
        params={"metric": "width", "comparator": "INVALID", "threshold": 640},
        branches={"then": [], "else": []},
    )
    code_fb, _ = export_pipeline_to_python([step_fallback])
    assert "> 640.0:" in code_fb
    ast.parse(code_fb)


def test_resilient_hex_color_handling():
    """Verify invalid or None hex color values fall back to default color without crashing."""
    step_none = PipelineStep(
        type="drawingoperations_drawrectangle",
        params={"rgbcolors_input": None},
    )
    code_none, _ = export_pipeline_to_python([step_none])
    ast.parse(code_none)

    step_invalid = PipelineStep(
        type="drawingoperations_drawcircle",
        params={"rgbcolors_input": "not_a_hex"},
    )
    code_invalid, _ = export_pipeline_to_python([step_invalid])
    ast.parse(code_invalid)
