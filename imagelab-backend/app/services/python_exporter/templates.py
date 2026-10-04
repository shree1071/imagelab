"""Operator code generation templates for ImageLab Python exporter.

Each template takes `params: dict` and returns a tuple:
(lines_of_code: list[str], needs_numpy: bool)
"""

from collections.abc import Callable

import cv2

from app.utils.color import hex_to_bgr

# Map interpolation strings to OpenCV constants
INTERPOLATION_MAP = {
    "NEAREST": "cv2.INTER_NEAREST",
    "LINEAR": "cv2.INTER_LINEAR",
    "CUBIC": "cv2.INTER_CUBIC",
    "AREA": "cv2.INTER_AREA",
    "LANCZOS4": "cv2.INTER_LANCZOS4",
}

# Flip code map
FLIP_MAP = {
    "X": 0,  # Vertical flip (around x-axis)
    "Y": 1,  # Horizontal flip (around y-axis)
    "BOTH": -1,  # Both axes
}

BORDER_TYPE_MAP = {
    "CONSTANT": "cv2.BORDER_CONSTANT",
    "REPLICATE": "cv2.BORDER_REPLICATE",
    "REFLECT": "cv2.BORDER_REFLECT",
    "WRAP": "cv2.BORDER_WRAP",
    "REFLECT_101": "cv2.BORDER_REFLECT_101",
}

MORPH_OP_MAP = {
    "ERODE": "cv2.MORPH_ERODE",
    "DILATE": "cv2.MORPH_DILATE",
    "OPEN": "cv2.MORPH_OPEN",
    "CLOSE": "cv2.MORPH_CLOSE",
    "GRADIENT": "cv2.MORPH_GRADIENT",
    "TOPHAT": "cv2.MORPH_TOPHAT",
    "BLACKHAT": "cv2.MORPH_BLACKHAT",
}

COLORMAP_MAP = {
    "AUTUMN": "cv2.COLORMAP_AUTUMN",
    "BONE": "cv2.COLORMAP_BONE",
    "JET": "cv2.COLORMAP_JET",
    "WINTER": "cv2.COLORMAP_WINTER",
    "RAINBOW": "cv2.COLORMAP_RAINBOW",
    "OCEAN": "cv2.COLORMAP_OCEAN",
    "SUMMER": "cv2.COLORMAP_SUMMER",
    "SPRING": "cv2.COLORMAP_SPRING",
    "COOL": "cv2.COLORMAP_COOL",
    "HSV": "cv2.COLORMAP_HSV",
    "PINK": "cv2.COLORMAP_PINK",
    "HOT": "cv2.COLORMAP_HOT",
    "PARULA": "cv2.COLORMAP_PARULA",
    "MAGMA": "cv2.COLORMAP_MAGMA",
    "INFERNO": "cv2.COLORMAP_INFERNO",
    "PLASMA": "cv2.COLORMAP_PLASMA",
    "VIRIDIS": "cv2.COLORMAP_VIRIDIS",
    "CIVIDIS": "cv2.COLORMAP_CIVIDIS",
    "TWILIGHT": "cv2.COLORMAP_TWILIGHT",
    "TWILIGHT_SHIFTED": "cv2.COLORMAP_TWILIGHT_SHIFTED",
    "TURBO": "cv2.COLORMAP_TURBO",
    "DEEPGREEN": "cv2.COLORMAP_DEEPGREEN",
}

DIST_TYPE_MAP = {
    "DIST_L1": "cv2.DIST_L1",
    "DIST_L2": "cv2.DIST_L2",
    "DIST_C": "cv2.DIST_C",
}

THRESH_TYPE_MAP = {
    "threshold_binary": "cv2.THRESH_BINARY",
    "threshold_binary_inv": "cv2.THRESH_BINARY_INV",
    "threshold_trunc": "cv2.THRESH_TRUNC",
    "threshold_tozero": "cv2.THRESH_TOZERO",
    "threshold_tozero_inv": "cv2.THRESH_TOZERO_INV",
    "BINARY": "cv2.THRESH_BINARY",
    "BINARY_INV": "cv2.THRESH_BINARY_INV",
    "TRUNC": "cv2.THRESH_TRUNC",
    "TOZERO": "cv2.THRESH_TOZERO",
    "TOZERO_INV": "cv2.THRESH_TOZERO_INV",
}


def _get_bgr_color(params: dict, key: str = "rgbcolors_input", default: str = "#2828cc") -> tuple[int, int, int]:
    raw = params.get(key)
    if not isinstance(raw, str) or not raw:
        raw = default
    try:
        return hex_to_bgr(raw)
    except Exception:
        return hex_to_bgr(default)


# Template generator handlers
OPERATOR_TEMPLATES: dict[str, Callable[[dict, str], tuple[list[str], bool]]] = {}


def register(name: str):
    def decorator(fn: Callable[[dict, str], tuple[list[str], bool]]):
        OPERATOR_TEMPLATES[name] = fn
        return fn

    return decorator


# ── Basic IO ──────────────────────────────────────────────────────────────────
@register("basic_readimage")
def _basic_readimage(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"# {var} is already loaded from input path"], False


@register("basic_writeimage")
def _basic_writeimage(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"# {var} will be saved to output path"], False


# ── Conversions ───────────────────────────────────────────────────────────────
@register("imageconvertions_grayimage")
def _grayimage(params: dict, var: str) -> tuple[list[str], bool]:
    code = [
        f"if {var}.ndim == 3:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
    ]
    return code, False


@register("imageconvertions_bgrtohsv")
def _bgrtohsv(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.cvtColor({var}, cv2.COLOR_BGR2HSV)"], False


@register("imageconvertions_hsvtobgr")
def _hsvtobgr(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.cvtColor({var}, cv2.COLOR_HSV2BGR)"], False


@register("imageconvertions_bgrtolab")
def _bgrtolab(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.cvtColor({var}, cv2.COLOR_BGR2Lab)"], False


@register("imageconvertions_labtobgr")
def _labtobgr(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.cvtColor({var}, cv2.COLOR_Lab2BGR)"], False


@register("imageconvertions_bgrtoycrcb")
def _bgrtoycrcb(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.cvtColor({var}, cv2.COLOR_BGR2YCrCb)"], False


@register("imageconvertions_ycrcbtobgr")
def _ycrcbtobgr(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.cvtColor({var}, cv2.COLOR_YCrCb2BGR)"], False


@register("imageconvertions_invertimage")
def _invertimage(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.bitwise_not({var})"], False


@register("imageconvertions_brightnessandcontrast")
def _brightness_contrast(params: dict, var: str) -> tuple[list[str], bool]:
    brightness = int(params.get("brightnessValue", 0))
    contrast = float(params.get("contrastValue", 1.0))
    return [f"{var} = cv2.convertScaleAbs({var}, alpha={contrast}, beta={brightness})"], False


@register("imageconvertions_histogramequalization")
def _hist_eq(params: dict, var: str) -> tuple[list[str], bool]:
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.equalizeHist({var})",
        f"elif {var}.ndim == 3 and {var}.shape[2] in (3, 4):",
        f"    _ycrcb = cv2.cvtColor({var}, cv2.COLOR_BGR2YCrCb if {var}.shape[2] == 3 else cv2.COLOR_BGRA2YCrCb)",
        "    _ycrcb[:, :, 0] = cv2.equalizeHist(_ycrcb[:, :, 0])",
        f"    {var} = cv2.cvtColor(_ycrcb, cv2.COLOR_YCrCb2BGR if {var}.shape[2] == 3 else cv2.COLOR_YCrCb2BGRA)",
    ]
    return code, False


@register("imageconvertions_clahe")
def _clahe(params: dict, var: str) -> tuple[list[str], bool]:
    clip_limit = float(params.get("clipLimit", 2.0))
    grid_x = int(params.get("tileGridSizeX", 8))
    grid_y = int(params.get("tileGridSizeY", 8))
    code = [
        f"_clahe = cv2.createCLAHE(clipLimit={clip_limit}, tileGridSize=({grid_x}, {grid_y}))",
        f"if {var}.ndim == 2:",
        f"    {var} = _clahe.apply({var})",
        f"elif {var}.ndim == 3 and {var}.shape[2] in (3, 4):",
        f"    _ycrcb = cv2.cvtColor({var}, cv2.COLOR_BGR2YCrCb if {var}.shape[2] == 3 else cv2.COLOR_BGRA2YCrCb)",
        "    _ycrcb[:, :, 0] = _clahe.apply(_ycrcb[:, :, 0])",
        f"    {var} = cv2.cvtColor(_ycrcb, cv2.COLOR_YCrCb2BGR if {var}.shape[2] == 3 else cv2.COLOR_YCrCb2BGRA)",
    ]
    return code, False


@register("imageconvertions_channelsplit")
def _channel_split(params: dict, var: str) -> tuple[list[str], bool]:
    channel_str = str(params.get("channel", "RED")).upper()
    ch_idx = 2 if channel_str in ("RED", "R") else (1 if channel_str in ("GREEN", "G") else 0)
    code = [
        f"if {var}.ndim == 3 and {var}.shape[2] >= 3:",
        f"    {var} = {var}[:, :, {ch_idx}]",
    ]
    return code, False


@register("imageconvertions_colormaps")
def _colormaps(params: dict, var: str) -> tuple[list[str], bool]:
    cm_name = str(params.get("type", "HOT")).upper()
    cm_const = COLORMAP_MAP.get(cm_name, "cv2.COLORMAP_HOT")
    return [f"{var} = cv2.applyColorMap({var}, {cm_const})"], False


@register("imageconvertions_graytobinary")
def _graytobinary(params: dict, var: str) -> tuple[list[str], bool]:
    threshold = float(params.get("thresholdValue", 127))
    max_val = float(params.get("maxValue", 255))
    code = [
        f"if {var}.ndim == 3:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
        f"_, {var} = cv2.threshold({var}, {threshold}, {max_val}, cv2.THRESH_BINARY)",
    ]
    return code, False


@register("imageconvertions_colortobinary")
def _colortobinary(params: dict, var: str) -> tuple[list[str], bool]:
    threshold = float(params.get("thresholdValue", 0))
    max_val = float(params.get("maxValue", 255))
    thresh_type_raw = str(params.get("thresholdType", "threshold_binary"))
    thresh_type = THRESH_TYPE_MAP.get(thresh_type_raw, "cv2.THRESH_BINARY")
    code = [
        f"if {var}.ndim == 3:",
        f"    _gray = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
        "else:",
        f"    _gray = {var}",
        f"_, {var} = cv2.threshold(_gray, {threshold}, {max_val}, {thresh_type})",
    ]
    return code, False


# ── Blurring ──────────────────────────────────────────────────────────────────
@register("blurring_applyblur")
def _blur(params: dict, var: str) -> tuple[list[str], bool]:
    w = int(params.get("widthSize", 3))
    h = int(params.get("heightSize", 3))
    return [f"{var} = cv2.blur({var}, ({w}, {h}))"], False


@register("blurring_applygaussianblur")
def _gaussian_blur(params: dict, var: str) -> tuple[list[str], bool]:
    w = int(params.get("widthSize", 1))
    h = int(params.get("heightSize", 1))
    # GaussianBlur requires odd kernel dimensions
    w = w if w % 2 != 0 else w + 1
    h = h if h % 2 != 0 else h + 1
    return [f"{var} = cv2.GaussianBlur({var}, ({w}, {h}), 0)"], False


@register("blurring_applymedianblur")
def _median_blur(params: dict, var: str) -> tuple[list[str], bool]:
    k = int(params.get("kernelSize", 3))
    k = k if k % 2 != 0 else k + 1
    return [f"{var} = cv2.medianBlur({var}, {k})"], False


# ── Filtering ─────────────────────────────────────────────────────────────────
@register("filtering_bilateral")
def _bilateral(params: dict, var: str) -> tuple[list[str], bool]:
    d = int(params.get("filterSize", 5))
    sigma_c = float(params.get("sigmaColor", 75))
    sigma_s = float(params.get("sigmaSpace", 75))
    return [f"{var} = cv2.bilateralFilter({var}, {d}, {sigma_c}, {sigma_s})"], False


@register("filtering_boxfilter")
def _boxfilter(params: dict, var: str) -> tuple[list[str], bool]:
    w = int(params.get("width", 3))
    h = int(params.get("height", 3))
    depth = int(params.get("depth", -1))
    return [f"{var} = cv2.boxFilter({var}, {depth}, ({w}, {h}))"], False


@register("filtering_cannyedge")
def _canny(params: dict, var: str) -> tuple[list[str], bool]:
    t1 = float(params.get("threshold1", 100))
    t2 = float(params.get("threshold2", 200))
    aperture = int(params.get("apertureSize", 3))
    code = [
        f"if {var}.ndim == 3:",
        f"    _gray = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
        "else:",
        f"    _gray = {var}",
        "if _gray.dtype != np.uint8:",
        "    _gray = np.clip(_gray, 0, 255).astype(np.uint8)",
        f"_edges = cv2.Canny(_gray, {t1}, {t2}, apertureSize={aperture})",
        f"{var} = cv2.cvtColor(_edges, cv2.COLOR_GRAY2BGR)",
    ]
    return code, True


@register("filtering_dilation")
def _dilate(params: dict, var: str) -> tuple[list[str], bool]:
    ksize = int(params.get("kernelSize", 5))
    iters = int(params.get("iteration", 1))
    code = [
        f"_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))",
        f"{var} = cv2.dilate({var}, _kernel, iterations={iters})",
    ]
    return code, False


@register("filtering_erosion")
def _erode(params: dict, var: str) -> tuple[list[str], bool]:
    ksize = int(params.get("kernelSize", 5))
    iters = int(params.get("iteration", 1))
    code = [
        f"_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))",
        f"{var} = cv2.erode({var}, _kernel, iterations={iters})",
    ]
    return code, False


@register("filtering_morphological")
def _morph(params: dict, var: str) -> tuple[list[str], bool]:
    morph_name = str(params.get("type", "TOPHAT")).upper()
    op = MORPH_OP_MAP.get(morph_name, "cv2.MORPH_TOPHAT")
    ksize = int(params.get("kernelSize", 5))
    iters = int(params.get("iteration", 1))
    code = [
        f"_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))",
        f"{var} = cv2.morphologyEx({var}, {op}, _kernel, iterations={iters})",
    ]
    return code, False


@register("filtering_pyramiddown")
def _pyrdown(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.pyrDown({var})"], False


@register("filtering_pyramidup")
def _pyrup(params: dict, var: str) -> tuple[list[str], bool]:
    return [f"{var} = cv2.pyrUp({var})"], False


@register("filtering_sharpen")
def _sharpen(params: dict, var: str) -> tuple[list[str], bool]:
    strength = float(params.get("strength", 1.0))
    code = [
        f"_kernel = np.array([[0, -1, 0], [-1, 4 + {strength}, -1], [0, -1, 0]], dtype=np.float32)",
        f"{var} = cv2.filter2D({var}, -1, _kernel)",
    ]
    return code, True


@register("filtering_gaborfilter")
def _gabor(params: dict, var: str) -> tuple[list[str], bool]:
    ksize = int(params.get("kernelSize", 21))
    sigma = float(params.get("sigma", 5.0))
    theta = float(params.get("theta", 0.0))
    lambd = float(params.get("lambda_", 10.0))
    gamma = float(params.get("gamma", 0.5))
    code = [
        f"_gabor = cv2.getGaborKernel(({ksize}, {ksize}), {sigma}, {theta}, {lambd}, {gamma}, 0, ktype=cv2.CV_32F)",
        f"_filtered = cv2.filter2D({var}, cv2.CV_32F, _gabor)",
        f"{var} = np.clip(_filtered, 0, 255).astype(np.uint8)",
    ]
    return code, True


@register("filtering_contourdetection")
def _contours(params: dict, var: str) -> tuple[list[str], bool]:
    color = _get_bgr_color(params, "rgbcolors_input", "#00ff00")
    thickness = int(params.get("thickness", 2))
    mode_str = str(params.get("mode", "EXTERNAL")).upper()
    mode = "cv2.RETR_TREE" if mode_str == "TREE" else "cv2.RETR_EXTERNAL"
    method_str = str(params.get("method", "SIMPLE")).upper()
    method = "cv2.CHAIN_APPROX_NONE" if method_str == "NONE" else "cv2.CHAIN_APPROX_SIMPLE"
    code = [
        f"if {var}.ndim == 3:",
        f"    _gray = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
        "else:",
        f"    _gray = {var}",
        "if _gray.dtype != np.uint8:",
        "    _gray = np.clip(_gray, 0, 255).astype(np.uint8)",
        f"_contours, _ = cv2.findContours(_gray, {mode}, {method})",
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"cv2.drawContours({var}, _contours, -1, {color}, {thickness})",
    ]
    return code, True


# ── Drawing ───────────────────────────────────────────────────────────────────
@register("drawingoperations_drawline")
def _drawline(params: dict, var: str) -> tuple[list[str], bool]:
    x1 = int(params.get("starting_point_x1", 0))
    y1 = int(params.get("starting_point_y1", 0))
    x2 = int(params.get("ending_point_x", 0))
    y2 = int(params.get("ending_point_y", 0))
    color = _get_bgr_color(params)
    thickness = int(params.get("thickness", 2))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"cv2.line({var}, ({x1}, {y1}), ({x2}, {y2}), {color}, {thickness})",
    ]
    return code, False


@register("drawingoperations_drawcircle")
def _drawcircle(params: dict, var: str) -> tuple[list[str], bool]:
    cx = int(params.get("center_point_x", 0))
    cy = int(params.get("center_point_y", 0))
    radius = int(params.get("radius", 5))
    color = _get_bgr_color(params)
    thickness = int(params.get("thickness", 2))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"cv2.circle({var}, ({cx}, {cy}), {radius}, {color}, {thickness})",
    ]
    return code, False


@register("drawingoperations_drawellipse")
def _drawellipse(params: dict, var: str) -> tuple[list[str], bool]:
    cx = int(params.get("center_point_x", 0))
    cy = int(params.get("center_point_y", 0))
    width = int(params.get("width", 0))
    height = int(params.get("height", 0))
    angle = int(params.get("angle", 0))
    color = _get_bgr_color(params)
    thickness = int(params.get("thickness", 2))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"cv2.ellipse({var}, ({cx}, {cy}), ({width}, {height}), {angle}, 0, 360, {color}, {thickness})",
    ]
    return code, False


@register("drawingoperations_drawrectangle")
def _drawrect(params: dict, var: str) -> tuple[list[str], bool]:
    x1 = int(params.get("starting_point_x", 0))
    y1 = int(params.get("starting_point_y", 0))
    x2 = int(params.get("ending_point_x", 0))
    y2 = int(params.get("ending_point_y", 0))
    color = _get_bgr_color(params)
    thickness = int(params.get("thickness", 2))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"cv2.rectangle({var}, ({x1}, {y1}), ({x2}, {y2}), {color}, {thickness})",
    ]
    return code, False


@register("drawingoperations_drawarrowline")
def _drawarrow(params: dict, var: str) -> tuple[list[str], bool]:
    x1 = int(params.get("starting_point_x", 0))
    y1 = int(params.get("starting_point_y", 0))
    x2 = int(params.get("ending_point_x", 0))
    y2 = int(params.get("ending_point_y", 0))
    color = _get_bgr_color(params)
    thickness = int(params.get("thickness", 2))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"cv2.arrowedLine({var}, ({x1}, {y1}), ({x2}, {y2}), {color}, {thickness})",
    ]
    return code, False


@register("drawingoperations_drawtext")
def _drawtext(params: dict, var: str) -> tuple[list[str], bool]:
    text = str(params.get("draw_text", "Image Lab"))
    x = int(params.get("starting_point_x", 0))
    y = int(params.get("starting_point_y", 50))
    scale = float(params.get("scale", 1.0))
    color = _get_bgr_color(params)
    thickness = int(params.get("thickness", 2))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"cv2.putText({var}, {text!r}, ({x}, {y}), cv2.FONT_HERSHEY_SIMPLEX, {scale}, {color}, {thickness})",
    ]
    return code, False


# ── Geometric ─────────────────────────────────────────────────────────────────
@register("geometric_reflectimage")
def _reflect(params: dict, var: str) -> tuple[list[str], bool]:
    flip_type = str(params.get("type", "X"))
    flip_code = FLIP_MAP.get(flip_type, 0)
    return [f"{var} = cv2.flip({var}, {flip_code})"], False


@register("geometric_cropimage")
def _crop(params: dict, var: str) -> tuple[list[str], bool]:
    x1 = int(params.get("x1", 0))
    y1 = int(params.get("y1", 0))
    x2 = params.get("x2")
    y2 = params.get("y2")
    x2_str = str(int(x2)) if x2 is not None else f"{var}.shape[1]"
    y2_str = str(int(y2)) if y2 is not None else f"{var}.shape[0]"
    return [f"{var} = {var}[{y1}:{y2_str}, {x1}:{x2_str}]"], False


@register("geometric_resizeimage")
def _resize(params: dict, var: str) -> tuple[list[str], bool]:
    w = int(params.get("width", 300))
    h = int(params.get("height", 300))
    interp_name = str(params.get("interpolation", "LINEAR")).upper()
    interp = INTERPOLATION_MAP.get(interp_name, "cv2.INTER_LINEAR")
    return [f"{var} = cv2.resize({var}, ({w}, {h}), interpolation={interp})"], False


@register("geometric_rotateimage")
def _rotate(params: dict, var: str) -> tuple[list[str], bool]:
    angle = float(params.get("angle", 90.0))
    scale = float(params.get("scale", 1.0))
    code = [
        f"_h, _w = {var}.shape[:2]",
        f"_m = cv2.getRotationMatrix2D((_w / 2, _h / 2), {angle}, {scale})",
        f"{var} = cv2.warpAffine({var}, _m, (_w, _h))",
    ]
    return code, False


@register("geometric_scaleimage")
def _scale(params: dict, var: str) -> tuple[list[str], bool]:
    fx = float(params.get("fx", 1.0))
    fy = float(params.get("fy", 1.0))
    interp_name = str(params.get("interpolation", "LINEAR")).upper()
    interp = INTERPOLATION_MAP.get(interp_name, "cv2.INTER_LINEAR")
    return [f"{var} = cv2.resize({var}, (0, 0), fx={fx}, fy={fy}, interpolation={interp})"], False


@register("geometric_affineimage")
def _affine(params: dict, var: str) -> tuple[list[str], bool]:
    tx = float(params.get("translate_x", 0.0))
    ty = float(params.get("translate_y", 0.0))
    code = [
        f"_h, _w = {var}.shape[:2]",
        f"_m = np.float32([[1, 0, {tx}], [0, 1, {ty}]])",
        f"{var} = cv2.warpAffine({var}, _m, (_w, _h))",
    ]
    return code, True


# ── Thresholding ──────────────────────────────────────────────────────────────
@register("thresholding_applythreshold")
def _threshold(params: dict, var: str) -> tuple[list[str], bool]:
    thresh = float(params.get("thresholdValue", 0.0))
    max_val = float(params.get("maxValue", 255.0))
    return [f"_, {var} = cv2.threshold({var}, {thresh}, {max_val}, cv2.THRESH_BINARY)"], False


@register("thresholding_adaptivethreshold")
def _adaptive_threshold(params: dict, var: str) -> tuple[list[str], bool]:
    max_val = float(params.get("maxValue", 255.0))
    block_size = int(params.get("blockSize", 3))
    block_size = block_size if block_size % 2 != 0 else block_size + 1
    c_val = float(params.get("cValue", 2.0))
    method_str = str(params.get("adaptiveMethod", "GAUSSIAN")).upper()
    method = "cv2.ADAPTIVE_THRESH_MEAN_C" if method_str == "MEAN" else "cv2.ADAPTIVE_THRESH_GAUSSIAN_C"
    code = [
        f"if {var}.ndim == 3:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
        f"{var} = cv2.adaptiveThreshold({var}, {max_val}, {method}, cv2.THRESH_BINARY, {block_size}, {c_val})",
    ]
    return code, False


@register("thresholding_otsuthreshold")
def _otsu(params: dict, var: str) -> tuple[list[str], bool]:
    max_val = float(params.get("maxValue", 255.0))
    code = [
        f"if {var}.ndim == 3:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
        f"_, {var} = cv2.threshold({var}, 0, {max_val}, cv2.THRESH_BINARY + cv2.THRESH_OTSU)",
    ]
    return code, False


@register("thresholding_applyborders")
def _borders(params: dict, var: str) -> tuple[list[str], bool]:
    top = int(params.get("borderTop", 0))
    bottom = int(params.get("borderBottom", 0))
    left = int(params.get("borderLeft", 0))
    right = int(params.get("borderRight", 0))
    all_sides = params.get("border_all_sides")
    if all_sides is not None:
        val = int(all_sides)
        top, bottom, left, right = val, val, val, val
    btype_str = str(params.get("border_type", "CONSTANT")).upper()
    btype = BORDER_TYPE_MAP.get(btype_str, "cv2.BORDER_CONSTANT")
    color = _get_bgr_color(params, default="#000000")
    return [f"{var} = cv2.copyMakeBorder({var}, {top}, {bottom}, {left}, {right}, {btype}, value={color})"], False


@register("border_for_all")
def _border_all(params: dict, var: str) -> tuple[list[str], bool]:
    return ["# border_for_all configuration handled in apply borders"], False


@register("border_each_side")
def _border_each(params: dict, var: str) -> tuple[list[str], bool]:
    return ["# border_each_side configuration handled in apply borders"], False


# ── Sobel Derivatives & Laplacian ─────────────────────────────────────────────
@register("sobelderivatives_soblederivate")
def _sobel(params: dict, var: str) -> tuple[list[str], bool]:
    ddepth = int(params.get("ddepth", cv2.CV_64F))
    direction = str(params.get("type", "HORIZONTAL")).upper()
    dx, dy = (1, 0) if direction == "HORIZONTAL" else (0, 1)
    ksize = int(params.get("ksize", 3))
    return [f"{var} = cv2.Sobel({var}, {ddepth}, {dx}, {dy}, ksize={ksize})"], False


@register("sobelderivatives_scharrderivate")
def _scharr(params: dict, var: str) -> tuple[list[str], bool]:
    ddepth = int(params.get("ddepth", cv2.CV_64F))
    direction = str(params.get("type", "HORIZONTAL")).upper()
    dx, dy = (1, 0) if direction == "HORIZONTAL" else (0, 1)
    return [f"{var} = cv2.Scharr({var}, {ddepth}, {dx}, {dy})"], False


@register("sobelderivatives_prewittoperator")
def _prewitt(params: dict, var: str) -> tuple[list[str], bool]:
    code = [
        "_kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)",
        "_ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)",
        f"_gx = cv2.filter2D({var}, cv2.CV_32F, _kx)",
        f"_gy = cv2.filter2D({var}, cv2.CV_32F, _ky)",
        f"{var} = cv2.convertScaleAbs(cv2.magnitude(_gx, _gy))",
    ]
    return code, True


@register("laplacian")
@register("transformation_laplacian")
def _laplacian(params: dict, var: str) -> tuple[list[str], bool]:
    ksize = int(params.get("ksize", 1))
    ddepth = int(params.get("ddepth", cv2.CV_64F))
    return [f"{var} = cv2.Laplacian({var}, {ddepth}, ksize={ksize})"], False


# ── Transformation ────────────────────────────────────────────────────────────
@register("transformation_distance")
def _distance_transform(params: dict, var: str) -> tuple[list[str], bool]:
    dist_type_name = str(params.get("type", "DIST_L2")).upper()
    dist_type = DIST_TYPE_MAP.get(dist_type_name, "cv2.DIST_L2")
    mask_size = int(params.get("maskSize", 5))
    code = [
        f"if {var}.ndim == 3:",
        f"    _gray = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY if {var}.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)",
        "else:",
        f"    _gray = {var}",
        "_, _bin = cv2.threshold(_gray, 127, 255, cv2.THRESH_BINARY)",
        f"_dt = cv2.distanceTransform(_bin, {dist_type}, {mask_size})",
        f"{var} = cv2.normalize(_dt, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)",
    ]
    return code, False


# ── Augmentation ──────────────────────────────────────────────────────────────
@register("augmentation_gaussiannoise")
def _gaussian_noise(params: dict, var: str) -> tuple[list[str], bool]:
    mean = float(params.get("mean", 0.0))
    std = float(params.get("std", 25.0))
    code = [
        f"_noise = np.random.normal({mean}, {std}, {var}.shape).astype(np.float32)",
        f"{var} = np.clip({var}.astype(np.float32) + _noise, 0, 255).astype(np.uint8)",
    ]
    return code, True


@register("augmentation_saltpeppernoise")
def _salt_pepper(params: dict, var: str) -> tuple[list[str], bool]:
    amount = float(params.get("amount", 0.05))
    s_vs_p = float(params.get("s_vs_p", 0.5))
    code = [
        f"_noisy = {var}.copy()",
        f"_num_salt = int(np.ceil({amount} * {var}.size * {s_vs_p}))",
        f"_num_pepper = int(np.ceil({amount} * {var}.size * (1.0 - {s_vs_p})))",
        f"_coords_s = [np.random.randint(0, _d, _num_salt) for _d in {var}.shape[:2]]",
        "_noisy[tuple(_coords_s)] = 255",
        f"_coords_p = [np.random.randint(0, _d, _num_pepper) for _d in {var}.shape[:2]]",
        "_noisy[tuple(_coords_p)] = 0",
        f"{var} = _noisy",
    ]
    return code, True


@register("augmentation_sepiafilter")
def _sepia(params: dict, var: str) -> tuple[list[str], bool]:
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        "_sepia_matrix = np.array([",
        "    [0.272, 0.534, 0.131],",
        "    [0.349, 0.686, 0.168],",
        "    [0.393, 0.769, 0.189]",
        "], dtype=np.float32)",
        f"{var} = cv2.transform({var}, _sepia_matrix)",
        f"{var} = np.clip({var}, 0, 255).astype(np.uint8)",
    ]
    return code, True


# ── Segmentation ──────────────────────────────────────────────────────────────
@register("segmentation_kmeans")
def _kmeans(params: dict, var: str) -> tuple[list[str], bool]:
    k = int(params.get("k", 3))
    max_iter = int(params.get("max_iter", 100))
    eps = float(params.get("epsilon", 0.2))
    attempts = int(params.get("attempts", 3))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"_data = {var}.reshape((-1, 3)).astype(np.float32)",
        f"_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, {max_iter}, {eps})",
        f"_, _labels, _centers = cv2.kmeans(_data, {k}, None, _criteria, {attempts}, cv2.KMEANS_RANDOM_CENTERS)",
        f"{var} = _centers[_labels.flatten()].reshape({var}.shape).astype(np.uint8)",
    ]
    return code, True


@register("segmentation_meanshift")
def _meanshift(params: dict, var: str) -> tuple[list[str], bool]:
    sp = int(params.get("sp", 21))
    sr = int(params.get("sr", 51))
    max_level = int(params.get("maxLevel", 1))
    code = [
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        f"{var} = cv2.pyrMeanShiftFiltering({var}, sp={sp}, sr={sr}, maxLevel={max_level})",
    ]
    return code, False


@register("segmentation_watershed")
def _watershed(params: dict, var: str) -> tuple[list[str], bool]:
    fg_thresh = float(params.get("foreground_threshold", 0.5))
    code = [
        f"_bgr = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR) if {var}.ndim == 2 else {var}.copy()",
        "_gray = cv2.cvtColor(_bgr, cv2.COLOR_BGR2GRAY)",
        "_, _thresh = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)",
        "_kernel = np.ones((3, 3), np.uint8)",
        "_opening = cv2.morphologyEx(_thresh, cv2.MORPH_OPEN, _kernel, iterations=2)",
        "_sure_bg = cv2.dilate(_opening, _kernel, iterations=3)",
        "_dist = cv2.distanceTransform(_opening, cv2.DIST_L2, 5)",
        f"_, _sure_fg = cv2.threshold(_dist, {fg_thresh} * _dist.max(), 255, 0)",
        "_sure_fg = np.uint8(_sure_fg)",
        "_unknown = cv2.subtract(_sure_bg, _sure_fg)",
        "_, _markers = cv2.connectedComponents(_sure_fg)",
        "_markers = _markers + 1",
        "_markers[_unknown == 255] = 0",
        "_markers = cv2.watershed(_bgr, _markers)",
        "_bgr[_markers == -1] = [0, 0, 255]",
        f"{var} = _bgr",
    ]
    return code, True


# ── Detection ─────────────────────────────────────────────────────────────────
@register("detection_eyedetection")
def _eyedetection(params: dict, var: str) -> tuple[list[str], bool]:
    sf = float(params.get("scaleFactor", 1.1))
    mn = int(params.get("minNeighbors", 5))
    mw = int(params.get("minWidth", 10))
    mh = int(params.get("minHeight", 10))
    color = _get_bgr_color(params, default="#00ff00")
    thickness = int(params.get("thickness", 2))
    code = [
        "_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')",
        f"_gray = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY) if {var}.ndim == 3 else {var}",
        "_eq = cv2.equalizeHist(_gray)",
        f"_eyes = _eye_cascade.detectMultiScale(_eq, scaleFactor={sf}, minNeighbors={mn}, minSize=({mw}, {mh}))",
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        "for (_x, _y, _w, _h) in _eyes:",
        f"    cv2.rectangle({var}, (_x, _y), (_x + _w, _y + _h), {color}, {thickness})",
    ]
    return code, False


@register("detection_facedetection")
def _facedetection(params: dict, var: str) -> tuple[list[str], bool]:
    sf = float(params.get("scaleFactor", 1.1))
    mn = int(params.get("minNeighbors", 5))
    mw = int(params.get("minWidth", 30))
    mh = int(params.get("minHeight", 30))
    color = _get_bgr_color(params, default="#00ff00")
    thickness = int(params.get("thickness", 2))
    code = [
        "_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')",
        f"_gray = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY) if {var}.ndim == 3 else {var}",
        "_eq = cv2.equalizeHist(_gray)",
        f"_faces = _face_cascade.detectMultiScale(_eq, scaleFactor={sf}, minNeighbors={mn}, minSize=({mw}, {mh}))",
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        "for (_x, _y, _w, _h) in _faces:",
        f"    cv2.rectangle({var}, (_x, _y), (_x + _w, _y + _h), {color}, {thickness})",
    ]
    return code, False


@register("detection_smiledetection")
def _smiledetection(params: dict, var: str) -> tuple[list[str], bool]:
    sf = float(params.get("scaleFactor", 1.1))
    mn = int(params.get("minNeighbors", 5))
    mw = int(params.get("minWidth", 30))
    mh = int(params.get("minHeight", 30))
    color = _get_bgr_color(params, default="#00ff00")
    thickness = int(params.get("thickness", 2))
    draw_faces_raw = params.get("drawFaceBoxes", False)
    draw_faces = draw_faces_raw == "TRUE" if isinstance(draw_faces_raw, str) else bool(draw_faces_raw)
    code = [
        "_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')",
        "_smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')",
        f"_gray = cv2.cvtColor({var}, cv2.COLOR_BGR2GRAY) if {var}.ndim == 3 else {var}",
        "_eq = cv2.equalizeHist(_gray)",
        f"_faces = _face_cascade.detectMultiScale(_eq, scaleFactor={sf}, minNeighbors={mn}, minSize=({mw}, {mh}))",
        f"if {var}.ndim == 2:",
        f"    {var} = cv2.cvtColor({var}, cv2.COLOR_GRAY2BGR)",
        "for (_x, _y, _w, _h) in _faces:",
    ]
    if draw_faces:
        code.append(f"    cv2.rectangle({var}, (_x, _y), (_x + _w, _y + _h), (255, 100, 0), {thickness})")
    code.extend(
        [
            "    _roi_y = _y + int(_h * 0.6)",
            "    _smile_roi = _eq[_roi_y : _y + _h, _x : _x + _w]",
            (
                f"    _smiles = _smile_cascade.detectMultiScale("
                f"_smile_roi, scaleFactor={sf}, minNeighbors={mn}, minSize=({mw} // 2, {mh} // 2))"
            ),
            "    for (_sx, _sy, _sw, _sh) in _smiles:",
            (
                f"        cv2.rectangle("
                f"{var}, (_x + _sx, _roi_y + _sy), (_x + _sx + _sw, _roi_y + _sy + _sh), {color}, {thickness})"
            ),
        ]
    )
    return code, False


# ── Arithmetic ────────────────────────────────────────────────────────────────
@register("arithmetic_addweighted")
def _add_weighted(params: dict, var: str) -> tuple[list[str], bool]:
    alpha = float(params.get("alpha", 0.5))
    beta = float(params.get("beta", 1.0 - alpha))
    gamma = float(params.get("gamma", 0.0))
    return [f"{var} = cv2.addWeighted({var}, {alpha}, {var}, {beta}, {gamma})"], False
