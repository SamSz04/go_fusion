"""
HLO IR data classes for representing XLA High-Level Optimizer instructions.

These data classes model the core HLO concepts: shapes, instructions,
computations, and modules. They are used throughout the GO fusion system
for analysis, fusion simulation, and performance modeling.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class HloShape:
    """Shape descriptor for an HLO instruction output.

    Supports both simple tensor shapes like ``f32[1024,512]{1,0}`` and
    tuple shapes like ``(f32[1024,1024]{0,1}, s8[4194304]{0})``.

    Attributes:
        element_type: Scalar type string, e.g. "f32", "s32", "bf16", "pred".
            Empty string for tuple shapes.
        dimensions: Dimension sizes, e.g. [1024, 512] for a 2-D tensor.
            Empty for tuple shapes and scalars.
        layout: Optional minor-to-major layout ordering.
        is_tuple: Whether this is a tuple of sub-shapes.
        tuple_shapes: Sub-shapes when is_tuple is True.
    """
    element_type: str = ""
    dimensions: List[int] = field(default_factory=list)
    layout: Optional[List[int]] = None
    is_tuple: bool = False
    tuple_shapes: List["HloShape"] = field(default_factory=list)

    @property
    def rank(self) -> int:
        """Number of dimensions (0 for scalars and tuple shapes)."""
        return len(self.dimensions)

    @property
    def num_elements(self) -> int:
        """Total number of scalar elements in this shape."""
        if self.is_tuple:
            return sum(s.num_elements for s in self.tuple_shapes)
        if not self.dimensions:
            return 1
        result = 1
        for d in self.dimensions:
            result *= d
        return result

    @property
    def element_size_bytes(self) -> int:
        """Size in bytes of a single scalar element."""
        size_map = {
            "f32": 4, "s32": 4, "u32": 4,
            "f64": 8, "s64": 8, "u64": 8,
            "f16": 2, "bf16": 2,
            "s16": 2, "u16": 2,
            "s8": 1, "u8": 1,
            "pred": 1,
        }
        return size_map.get(self.element_type, 4)

    @property
    def total_bytes(self) -> int:
        """Total size in bytes of this tensor."""
        if self.is_tuple:
            return sum(s.total_bytes for s in self.tuple_shapes)
        return self.num_elements * self.element_size_bytes

    def __str__(self) -> str:
        if self.is_tuple:
            inner = ", ".join(str(s) for s in self.tuple_shapes)
            return f"({inner})"
        dims = ",".join(str(d) for d in self.dimensions)
        base = f"{self.element_type}[{dims}]" if self.dimensions else f"{self.element_type}[]"
        if self.layout is not None:
            layout_str = ",".join(str(x) for x in self.layout)
            base += "{" + layout_str + "}"
        return base


@dataclass
class HloInstruction:
    """A single HLO instruction (node in the dataflow graph).

    Attributes:
        name: Unique instruction name (without leading '%'),
            e.g. "add.309.0".
        opcode: Operation code, e.g. "add", "dot", "reduce",
            "custom-call", "fusion".
        shape: Output shape of this instruction.
        operand_names: Names (without '%') of operand instructions
            that define the dataflow edges into this node.
        attributes: Opcode-specific attributes parsed from the
            instruction line, e.g. {"custom_call_target": "__cublas$gemm",
            "kind": "kCustom", "calls": "computation_name", "index": "0",
            "direction": "LT", "dimensions": "{1}"}.
        metadata: Metadata key-value pairs from the metadata={...} block,
            e.g. {"op_name": "jit(.../add", "stack_frame_id": "58"}.
        is_root: Whether this instruction is the ROOT of its computation.
        computation_name: Name of the computation that owns this instruction.
    """
    name: str = ""
    opcode: str = ""
    shape: HloShape = field(default_factory=HloShape)
    operand_names: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    is_root: bool = False
    computation_name: str = ""

    # ---- convenience properties ----

    @property
    def is_parameter(self) -> bool:
        return self.opcode == "parameter"

    @property
    def is_constant(self) -> bool:
        return self.opcode == "constant"

    @property
    def is_fusion(self) -> bool:
        return self.opcode == "fusion"

    @property
    def is_custom_call(self) -> bool:
        return self.opcode == "custom-call"

    @property
    def is_gemm(self) -> bool:
        """True if this is a cuBLAS GEMM custom-call."""
        if self.opcode == "custom-call":
            target = self.attributes.get("custom_call_target", "")
            return "__cublas$gemm" in target
        return False

    @property
    def is_fusable(self) -> bool:
        """Whether this node can participate in operator fusion.

        Parameters, constants, get-tuple-element, and tuple ops are
        generally structural and not fusable on their own.
        """
        non_fusable = {"parameter", "constant", "get-tuple-element", "tuple"}
        return self.opcode not in non_fusable

    @property
    def fusion_kind(self) -> Optional[str]:
        """Return the fusion kind (e.g. 'kCustom') if this is a fusion op."""
        if self.opcode == "fusion":
            return self.attributes.get("kind")
        return None

    @property
    def called_computation(self) -> Optional[str]:
        """Name of the sub-computation called by this instruction, if any."""
        return self.attributes.get("calls") or self.attributes.get("to_apply")

    def __repr__(self) -> str:
        return (
            f"HloInstruction(name={self.name!r}, opcode={self.opcode!r}, "
            f"shape={self.shape})"
        )


@dataclass
class HloComputation:
    """An HLO computation (a named function containing instructions).

    Attributes:
        name: Computation name (without '%'), e.g. "main.27",
            "triton_softmax_computation.25".
        instructions: Ordered list of instructions in this computation.
        root_instruction: The ROOT instruction object, or None if not
            yet determined.
        is_entry: True if this is the ENTRY computation.
    """
    name: str = ""
    instructions: List[HloInstruction] = field(default_factory=list)
    root_instruction: Optional[HloInstruction] = None
    is_entry: bool = False

    def instruction_map(self) -> Dict[str, HloInstruction]:
        """Return a dict mapping instruction name -> HloInstruction."""
        return {inst.name: inst for inst in self.instructions}

    @property
    def parameter_names(self) -> List[str]:
        """Names of parameter instructions, sorted by parameter index."""
        params = [
            inst for inst in self.instructions if inst.opcode == "parameter"
        ]
        # Sort by parameter index if available, else by order of appearance.
        params.sort(
            key=lambda p: int(p.attributes.get("param_index", 0))
        )
        return [p.name for p in params]

    def __repr__(self) -> str:
        return (
            f"HloComputation(name={self.name!r}, "
            f"num_instructions={len(self.instructions)}, "
            f"is_entry={self.is_entry})"
        )


@dataclass
class HloModule:
    """Top-level HLO module containing all computations.

    Attributes:
        name: Module name, e.g. "jit_forward_pass".
        entry_computation: The ENTRY HloComputation object.
        computations: Dict mapping computation name to HloComputation.
        raw_metadata: Raw metadata tables (FileNames, FunctionNames,
            FileLocations, StackFrames) stored as dict of lists of
            raw text lines.
    """
    name: str = ""
    entry_computation: Optional[HloComputation] = None
    computations: Dict[str, HloComputation] = field(default_factory=dict)
    raw_metadata: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def num_computations(self) -> int:
        return len(self.computations)

    @property
    def entry_instruction_count(self) -> int:
        if self.entry_computation is None:
            return 0
        return len(self.entry_computation.instructions)

    def get_computation(self, name: str) -> Optional[HloComputation]:
        """Look up a computation by name (strips leading '%' if present)."""
        clean = name.lstrip("%")
        return self.computations.get(clean)

    def __repr__(self) -> str:
        return (
            f"HloModule(name={self.name!r}, "
            f"num_computations={self.num_computations}, "
            f"entry_instructions={self.entry_instruction_count})"
        )
