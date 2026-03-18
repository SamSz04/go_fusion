"""
HLO text format parser.

Parses XLA HLO textual IR files (as dumped by XLA's ``--xla_dump_to``) into
an ``HloModule`` data structure.  The parser is line-oriented and uses regex
for tokenisation.  It handles:

* Module header (``HloModule name, ...``)
* Metadata tables (FileNames, FunctionNames, FileLocations, StackFrames)
* Non-ENTRY sub-computations (``%name (...) -> type { ... }``)
* ENTRY computation (``ENTRY %name (...) -> type { ... }``)
* Instruction lines with shapes (simple and tuple), operands, attributes,
  metadata, and backend_config (with nested JSON braces).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .hlo_ir import HloComputation, HloInstruction, HloModule, HloShape


# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------

def parse_shape(shape_str: str) -> HloShape:
    """Parse a shape string like ``f32[1024,512]{1,0}`` into an HloShape.

    Also handles:
    * Scalars: ``f32[]``
    * Tuples: ``(f32[1024,1024]{0,1}, s8[4194304]{0})``
    """
    shape_str = shape_str.strip()

    # Tuple shape
    if shape_str.startswith("("):
        return _parse_tuple_shape(shape_str)

    # Simple shape
    return _parse_simple_shape(shape_str)


def _parse_tuple_shape(shape_str: str) -> HloShape:
    """Parse a tuple shape string like ``(f32[1024]{0}, s8[4194304]{0})``."""
    # Strip outer parens
    inner = shape_str[1:-1].strip()
    parts = _split_tuple_elements(inner)
    sub_shapes = [parse_shape(p.strip()) for p in parts]
    return HloShape(is_tuple=True, tuple_shapes=sub_shapes)


def _split_tuple_elements(s: str) -> List[str]:
    """Split comma-separated tuple elements, respecting nested parens."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


_SIMPLE_SHAPE_RE = re.compile(
    r"^(\w+)"            # element type, e.g. f32
    r"\[([^\]]*)\]"      # dimensions, e.g. 1024,512
    r"(?:\{([^\}]*)\})?" # optional layout, e.g. {1,0}
    r"$"
)

_SCALAR_SHAPE_RE = re.compile(
    r"^(\w+)\[\]$"       # scalar, e.g. f32[]
)


def _parse_simple_shape(shape_str: str) -> HloShape:
    """Parse a simple (non-tuple) shape string."""
    shape_str = shape_str.strip()

    # Scalar
    m = _SCALAR_SHAPE_RE.match(shape_str)
    if m:
        return HloShape(element_type=m.group(1), dimensions=[])

    m = _SIMPLE_SHAPE_RE.match(shape_str)
    if m:
        etype = m.group(1)
        dims_str = m.group(2)
        dims = [int(d) for d in dims_str.split(",") if d.strip()] if dims_str else []
        layout = None
        if m.group(3) is not None:
            layout = [int(x) for x in m.group(3).split(",") if x.strip()]
        return HloShape(element_type=etype, dimensions=dims, layout=layout)

    # Fallback: try to extract what we can
    return HloShape(element_type=shape_str)


# ---------------------------------------------------------------------------
# Shape extraction from instruction line
# ---------------------------------------------------------------------------

def _extract_shape_string(line: str) -> Tuple[str, int]:
    """Extract the full shape string starting after '= ' in an instruction.

    Returns (shape_string, end_index_in_line).  Handles tuples with nested
    parens.
    """
    eq_pos = line.find("= ")
    if eq_pos == -1:
        return "", 0

    start = eq_pos + 2
    # If the shape starts with '(' it is a tuple — track parens
    if start < len(line) and line[start] == "(":
        depth = 0
        i = start
        while i < len(line):
            if line[i] == "(":
                depth += 1
            elif line[i] == ")":
                depth -= 1
                if depth == 0:
                    return line[start:i + 1], i + 1
            i += 1
        # Fallback — unterminated tuple
        return line[start:], len(line)

    # Simple shape: runs until the next space
    end = line.find(" ", start)
    if end == -1:
        return line[start:], len(line)
    return line[start:end], end


# ---------------------------------------------------------------------------
# Instruction line parsing
# ---------------------------------------------------------------------------

def _parse_operands(operand_str: str) -> List[str]:
    """Parse the operand list ``(%a, %b, %c)`` into a list of names without %."""
    operand_str = operand_str.strip()
    if not operand_str:
        return []
    # Strip outer parens if present
    if operand_str.startswith("(") and operand_str.endswith(")"):
        operand_str = operand_str[1:-1]
    # Handle nested tuples in operands by splitting carefully
    names: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in operand_str:
        if ch in ("(", "{", "["):
            depth += 1
            current.append(ch)
        elif ch in (")", "}", "]"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(current).strip()
            if token.startswith("%"):
                names.append(token[1:])
            current = []
        else:
            current.append(ch)
    token = "".join(current).strip()
    if token.startswith("%"):
        names.append(token[1:])
    return names


def _parse_metadata(meta_str: str) -> Dict[str, str]:
    """Parse ``metadata={op_name="..." stack_frame_id=N}`` into a dict."""
    result: Dict[str, str] = {}
    if not meta_str:
        return result
    # Find pairs: key="value" or key=value
    for m in re.finditer(r'(\w+)="([^"]*)"', meta_str):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r'(\w+)=(\d+)', meta_str):
        if m.group(1) not in result:
            result[m.group(1)] = m.group(2)
    return result


def _extract_backend_config_end(line: str, start: int) -> int:
    """Find the end of a backend_config={...} block by tracking brace depth.

    ``start`` should point to the opening '{' of the JSON value.
    Returns the index one past the closing '}'.
    """
    depth = 0
    i = start
    while i < len(line):
        if line[i] == "{":
            depth += 1
        elif line[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        elif line[i] == '"':
            # Skip string literals to avoid counting braces inside them
            i += 1
            while i < len(line) and line[i] != '"':
                if line[i] == "\\":
                    i += 1  # skip escaped char
                i += 1
        i += 1
    return len(line)


def _split_attributes_respecting_braces(attrs_str: str) -> List[str]:
    """Split a comma-separated attribute string, respecting nested braces
    and quoted strings so that ``backend_config={...}`` stays as one token.
    """
    parts: List[str] = []
    depth = 0
    in_quote = False
    current: List[str] = []
    i = 0
    while i < len(attrs_str):
        ch = attrs_str[i]
        if ch == '"' and (i == 0 or attrs_str[i - 1] != "\\"):
            in_quote = not in_quote
            current.append(ch)
        elif not in_quote and ch in ("{", "(", "["):
            depth += 1
            current.append(ch)
        elif not in_quote and ch in ("}", ")", "]"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0 and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    remainder = "".join(current).strip()
    if remainder:
        parts.append(remainder)
    return parts


def parse_instruction_line(
    line: str, computation_name: str = ""
) -> Optional[HloInstruction]:
    """Parse a single HLO instruction line into an HloInstruction.

    Expected format (whitespace-stripped)::

        [ROOT] %name = <shape> opcode(operands), [attrs], metadata={...}

    Returns None if the line cannot be parsed as an instruction.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("}"):
        return None

    is_root = False
    if stripped.startswith("ROOT "):
        is_root = True
        stripped = stripped[5:].strip()

    # Must start with %
    if not stripped.startswith("%"):
        return None

    # Split on " = "
    eq_pos = stripped.find(" = ")
    if eq_pos == -1:
        return None

    name = stripped[1:eq_pos]  # drop leading %

    rest = stripped[eq_pos + 3:]  # after " = "

    # --- Extract shape ---
    # Shape can be a tuple "(type1, type2)" or a simple "type[dims]{layout}"
    if rest.startswith("("):
        # Tuple shape — find matching close paren
        shape_str, after_shape_pos = _extract_shape_string(stripped)
        # after_shape_pos is relative to stripped; re-derive from rest
        # Actually, let's re-extract from rest directly
        depth = 0
        i = 0
        while i < len(rest):
            if rest[i] == "(":
                depth += 1
            elif rest[i] == ")":
                depth -= 1
                if depth == 0:
                    shape_str = rest[:i + 1]
                    rest = rest[i + 1:].strip()
                    break
            i += 1
        else:
            shape_str = rest
            rest = ""
    else:
        # Simple shape: take up to the first space
        space_pos = rest.find(" ")
        if space_pos == -1:
            shape_str = rest
            rest = ""
        else:
            shape_str = rest[:space_pos]
            rest = rest[space_pos + 1:]

    shape = parse_shape(shape_str)

    # --- Extract opcode and operands ---
    # rest now starts with "opcode(operands), attrs..."
    # or "opcode(operands)" or "opcode(), ..."
    # Some opcodes like "get-tuple-element" have hyphens.
    # Opcode ends at the first '('
    operand_raw = ""  # raw text inside the operand parens
    paren_pos = rest.find("(")
    if paren_pos == -1:
        # No parens — e.g. "constant(0)" might have been split differently
        # Try to extract opcode from rest
        opcode = rest.split(",")[0].strip()
        operand_names: List[str] = []
        attrs_str = ""
    else:
        opcode = rest[:paren_pos].strip()
        # Find matching close paren for operands
        depth = 0
        i = paren_pos
        operand_end = len(rest)
        while i < len(rest):
            if rest[i] == "(":
                depth += 1
            elif rest[i] == ")":
                depth -= 1
                if depth == 0:
                    operand_end = i
                    break
            i += 1
        operand_raw = rest[paren_pos + 1:operand_end]
        operand_names = _parse_operands(operand_raw)
        rest = rest[operand_end + 1:].strip()
        if rest.startswith(","):
            rest = rest[1:].strip()
        attrs_str = rest

    # --- Parse attributes and metadata ---
    attributes: Dict[str, Any] = {}
    metadata: Dict[str, str] = {}

    if attrs_str:
        # Split on commas, respecting nested braces for backend_config
        attr_tokens = _split_attributes_respecting_braces(attrs_str)

        for token in attr_tokens:
            token = token.strip()
            if not token:
                continue

            # metadata={...}
            if token.startswith("metadata="):
                meta_inner = token[len("metadata="):]
                metadata = _parse_metadata(meta_inner)
                continue

            # backend_config={...} — store as raw string
            if token.startswith("backend_config="):
                attributes["backend_config"] = token[len("backend_config="):]
                continue

            # key=value pairs
            eq = token.find("=")
            if eq != -1:
                key = token[:eq].strip()
                val = token[eq + 1:].strip()
                # Strip quotes
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                # Strip leading % from computation names
                if val.startswith("%"):
                    val = val[1:]
                attributes[key] = val

    # Handle special opcodes that carry non-operand content in their parens.
    # For "parameter(N)" the N is the parameter index, not an operand.
    if opcode == "parameter":
        attributes["param_index"] = operand_raw.strip()
        operand_names = []

    # For "constant(value)" the value is a literal, not an operand.
    if opcode == "constant":
        operand_names = []

    # For iota(), no operands (dimension is in attributes).
    if opcode == "iota":
        operand_names = []

    inst = HloInstruction(
        name=name,
        opcode=opcode,
        shape=shape,
        operand_names=operand_names,
        attributes=attributes,
        metadata=metadata,
        is_root=is_root,
        computation_name=computation_name,
    )
    return inst


# ---------------------------------------------------------------------------
# Computation parsing
# ---------------------------------------------------------------------------

_COMPUTATION_HEADER_RE = re.compile(
    r"^(ENTRY\s+)?%(\S+)\s+\("
)


def _find_computation_boundaries(lines: List[str]) -> List[Tuple[int, int, str, bool]]:
    """Find (start_line, end_line, name, is_entry) for each computation.

    A computation starts with a line matching ``[ENTRY] %name (...) -> type {``
    and ends with a line that is just ``}``.
    """
    boundaries: List[Tuple[int, int, str, bool]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _COMPUTATION_HEADER_RE.match(line)
        if m:
            is_entry = m.group(1) is not None
            comp_name = m.group(2)
            start = i
            # Find the closing brace. The opening brace is on the header line.
            # We need to track brace depth because there may be nested
            # backend_config JSON.  However, computation braces are always at
            # column 0, so we look for a line that is just "}".
            j = i + 1
            while j < len(lines):
                if lines[j].strip() == "}":
                    boundaries.append((start, j, comp_name, is_entry))
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return boundaries


def _parse_computation(
    lines: List[str], start: int, end: int, name: str, is_entry: bool
) -> HloComputation:
    """Parse a computation from *lines[start+1 .. end-1]* (instruction lines)."""
    comp = HloComputation(name=name, is_entry=is_entry)
    instructions: List[HloInstruction] = []
    root: Optional[HloInstruction] = None

    for i in range(start + 1, end):
        line = lines[i]
        inst = parse_instruction_line(line, computation_name=name)
        if inst is not None:
            instructions.append(inst)
            if inst.is_root:
                root = inst

    comp.instructions = instructions
    comp.root_instruction = root
    return comp


# ---------------------------------------------------------------------------
# Metadata table parsing
# ---------------------------------------------------------------------------

_METADATA_SECTIONS = {"FileNames", "FunctionNames", "FileLocations", "StackFrames"}


def _parse_metadata_tables(lines: List[str]) -> Tuple[Dict[str, List[str]], int]:
    """Parse the metadata tables that appear after the module header line.

    Returns (metadata_dict, first_line_after_metadata).
    """
    metadata: Dict[str, List[str]] = {}
    current_section: Optional[str] = None
    i = 1  # skip line 0 (module header)

    while i < len(lines):
        line = lines[i].strip()

        # Check if this line starts a metadata section
        if line in _METADATA_SECTIONS:
            current_section = line
            metadata[current_section] = []
            i += 1
            continue

        # Check if we've hit a computation header (end of metadata area)
        if line.startswith("%") or line.startswith("ENTRY"):
            break

        # Check for empty line (section separator)
        if not line:
            if current_section is not None:
                current_section = None
            i += 1
            continue

        # Accumulate into current section
        if current_section is not None:
            metadata[current_section].append(line)

        i += 1

    return metadata, i


# ---------------------------------------------------------------------------
# Module-level parsing
# ---------------------------------------------------------------------------

_MODULE_HEADER_RE = re.compile(r"^HloModule\s+(\S+?)(?:,\s*|\s*$)")


def parse_hlo_file(filepath: str) -> HloModule:
    """Parse an HLO text file into an HloModule.

    Args:
        filepath: Path to the ``.hlo`` text file.

    Returns:
        Parsed HloModule with all computations, instructions, and metadata.
    """
    text = Path(filepath).read_text(encoding="utf-8")
    return parse_hlo_string(text)


def parse_hlo_string(text: str) -> HloModule:
    """Parse an HLO text string into an HloModule.

    This is the main entry point for parsing HLO IR. It handles:
    1. Module header line
    2. Metadata tables (FileNames, etc.)
    3. Sub-computations and the ENTRY computation

    Args:
        text: Full HLO text content.

    Returns:
        Parsed HloModule.
    """
    lines = text.splitlines()
    if not lines:
        return HloModule()

    # --- Module header ---
    header_match = _MODULE_HEADER_RE.match(lines[0])
    module_name = header_match.group(1) if header_match else "unknown"

    module = HloModule(name=module_name)

    # --- Metadata tables ---
    raw_metadata, _ = _parse_metadata_tables(lines)
    module.raw_metadata = raw_metadata

    # --- Computations ---
    boundaries = _find_computation_boundaries(lines)
    for start, end, comp_name, is_entry in boundaries:
        comp = _parse_computation(lines, start, end, comp_name, is_entry)
        module.computations[comp_name] = comp
        if is_entry:
            module.entry_computation = comp

    return module
