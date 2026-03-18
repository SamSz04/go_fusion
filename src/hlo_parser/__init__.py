"""
HLO parser package for the GO fusion system.

Provides:
* **Data classes** (``HloShape``, ``HloInstruction``, ``HloComputation``,
  ``HloModule``) for representing XLA HLO IR structures.
* **Parser** (``parse_hlo_file``, ``parse_hlo_string``) for reading HLO
  text format into the data-class representation.
* **Graph builder** (``build_graph``) for converting a parsed module into
  a ``torch_geometric.data.Data`` object suitable for GNN processing.
* **Feature encoder** (``encode_features``) for producing the 49-dim
  node feature vectors consumed by the GNN.
"""

from .hlo_ir import (
    HloShape,
    HloInstruction,
    HloComputation,
    HloModule,
)
from .parser import (
    parse_hlo_file,
    parse_hlo_string,
    parse_shape,
    parse_instruction_line,
)
from .graph_builder import build_graph
from .feature_encoder import (
    encode_features,
    encode_instruction,
    OPCODE_VOCAB,
    ETYPE_VOCAB,
    FEATURE_DIM,
)

__all__ = [
    # Data classes
    "HloShape",
    "HloInstruction",
    "HloComputation",
    "HloModule",
    # Parser
    "parse_hlo_file",
    "parse_hlo_string",
    "parse_shape",
    "parse_instruction_line",
    # Graph builder
    "build_graph",
    # Feature encoder
    "encode_features",
    "encode_instruction",
    "OPCODE_VOCAB",
    "ETYPE_VOCAB",
    "FEATURE_DIM",
]
