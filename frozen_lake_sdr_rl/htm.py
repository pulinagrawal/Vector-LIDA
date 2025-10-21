from __future__ import annotations

import numpy as np
import random
from typing import (
    List,
    Set,
    Dict,
    Tuple,
    Optional,
    Sequence,
    Union,
)

# Constants (Spatial Pooler)
CONNECTED_PERM = 0.5  # Permanence threshold for an input synapse to be considered connected
MIN_OVERLAP = 3       # Minimum overlap to be considered during inhibition
PERMANENCE_INC = 0.01
PERMANENCE_DEC = 0.01
DESIRED_LOCAL_ACTIVITY = 10

# Constants (Temporal Memory)
SEGMENT_ACTIVATION_THRESHOLD = 3   # Active connected distal synapses required for a segment to be active (prediction)
SEGMENT_LEARNING_THRESHOLD = 3     # Subset used to select best matching segment
INITIAL_DISTAL_PERM = 0.21         # Initial permanence for new distal synapses
NEW_SYNAPSE_MAX = 6                # New synapses to add when reinforcing a learning segment

input_space_size = 100


# Cell class represents a single cell in a column
class Cell:
    """Single cell within a column.

    Holds a (possibly empty) list of distal segments used for Temporal Memory.
    """

    segments: List['Segment']  # forward reference

    def __init__(self) -> None:
        self.segments = []

    def __repr__(self) -> str:
        return f"Cell(id={id(self)})"


class DistalSynapse:
    """Distal synapse referencing a source cell (Temporal Memory)."""

    source_cell: Cell
    permanence: float

    def __init__(self, source_cell: Cell, permanence: float) -> None:
        self.source_cell = source_cell
        self.permanence = permanence


class Segment:
    """Distal segment composed of synapses to previously active cells."""

    synapses: List[DistalSynapse]
    sequence_segment: bool

    def __init__(self, synapses: Optional[List[DistalSynapse]] = None) -> None:
        self.synapses = synapses if synapses is not None else []
        self.sequence_segment = False  # True if learned in a predictive context

    def active_synapses(self, active_cells: Set[Cell]) -> List[DistalSynapse]:
        """Return connected synapses whose source cell is active."""
        return [syn for syn in self.synapses if syn.source_cell in active_cells and syn.permanence > CONNECTED_PERM]

    def matching_synapses(self, prev_active_cells: Set[Cell]) -> List[DistalSynapse]:
        """Return synapses whose source cell was previously active (ignores permanence threshold)."""
        return [syn for syn in self.synapses if syn.source_cell in prev_active_cells]

# Synapse class represents a synapse which connects a segment to a source input
class Synapse:
    """Proximal synapse (input space) used by Spatial Pooler only."""

    source_input: int
    permanence: float

    def __init__(self, source_input: int, permanence: float) -> None:
        self.source_input = source_input
        self.permanence = permanence

# Column class represents a column in the HTM Region
class Column:
    position: Tuple[int, int]
    potential_synapses: List[Synapse]
    boost: float
    active_duty_cycle: float
    overlap_duty_cycle: float
    min_duty_cycle: float
    connected_synapses: List[Synapse]
    overlap: float
    cells: List[Cell]  # added dynamically after region init

    def __init__(self, potential_synapses: List[Synapse], position: Tuple[int, int]):
        self.position = position
        self.potential_synapses = potential_synapses
        self.boost = 1.0
        self.active_duty_cycle = 0.0
        self.overlap_duty_cycle = 0.0
        self.min_duty_cycle = 0.01
        self.connected_synapses = [s for s in potential_synapses if s.permanence > CONNECTED_PERM]
        self.overlap = 0.0

    def compute_overlap(self, input_vector: np.ndarray) -> None:
        """Compute overlap with current binary input vector and apply boost."""
        overlap = sum(1 for s in self.connected_synapses if input_vector[s.source_input])
        self.overlap = float(overlap * self.boost) if overlap >= MIN_OVERLAP else 0.0
        print(f"Column at position {self.position} has overlap: {self.overlap}")
        
print("Starting the Temporal Pooler process...")        

class TemporalPooler:
    """Prototype Spatial Pooler + simplified Temporal Memory.

    Type hints approximate container contents; algorithmic logic is unchanged.
    """

    input_space_size: int
    columns: List[Column]
    cells_per_column: int
    active_cells: Dict[int, Set[Cell]]
    winner_cells: Dict[int, Set[Cell]]
    predictive_cells: Dict[int, Set[Cell]]
    learning_segments: Dict[int, Set[Segment]]
    negative_segments: Dict[int, Set[Segment]]
    field_ranges: Dict[str, Tuple[int, int]]
    field_order: List[str]
    column_field_map: Dict[Column, Optional[str]]

    # Input variants accepted by compute_active_columns
    InputField = Union[np.ndarray, Sequence[int]]
    InputComposite = Union[
        np.ndarray,
        Sequence[int],
        Sequence[InputField],
        Dict[str, InputField],
    ]

    def __init__(
        self,
    input_space_size: int,
    column_count: int,
    cells_per_column: int,
    initial_synapses_per_column: int,
    ) -> None:
        # Explicitly store configured input size (must be provided)
        self.input_space_size = int(input_space_size)
        self.column_count = column_count
        self.columns = self.initialize_region(input_space_size, column_count, initial_synapses_per_column)
        self.cells_per_column = cells_per_column
        for c in self.columns:
            c.cells = [Cell() for _ in range(cells_per_column)]

        # Temporal Memory state (time-indexed)
        self.active_cells = {}
        self.winner_cells = {}
        self.predictive_cells = {}
        self.learning_segments = {}
        self.negative_segments = {}
        # Multi-field metadata
        self.field_ranges = {}
        self.field_order = []
        self.column_field_map = {}
        # Internal time counter (replaces external t parameter)
        self.current_t = 0

    # Initializes the columns in the region
    def initialize_region(
        self,
        input_space_size: int,
        column_count: int,
        initial_synapses_per_column: int,
    ) -> List[Column]:
        columns: List[Column] = []
        grid_size = int(column_count ** 0.5)  # Assuming a square grid for simplicity
        for i in range(column_count):
            x = i % grid_size
            y = i // grid_size
            position = (x, y)
            potential_synapses = [
                Synapse(int(np.random.randint(input_space_size)), float(np.random.uniform(0.4, 0.6)))
                for _ in range(initial_synapses_per_column)
            ]
            columns.append(Column(potential_synapses, position))
        print(f"Initialized {len(columns)} columns with positions and potential synapses.")
        return columns
        

    # Computes the active columns after applying inhibition
    def _compute_active_columns_list(
        self,
        input_vector: InputComposite,
        inhibition_radius: float,
    ) -> List[Column]:
        """Internal: return list of active Column objects (legacy behavior)."""
        combined = self.combine_input_fields(input_vector)
        for c in self.columns:
            c.compute_overlap(combined)
        return self.inhibition(self.columns, inhibition_radius)

    def compute_active_columns(
        self,
        input_vector: InputComposite,
        inhibition_radius: float,
    ) -> np.ndarray:
        """Public interface: return binary vector (length = number of columns) indicating active columns.

        Internally we still compute the list of Column objects for algorithmic use.
        A 1 in position i means `self.columns[i]` is active this timestep.
        """
        active_list = self._compute_active_columns_list(input_vector, inhibition_radius)
        mask = self.columns_to_binary(active_list)
        print(f"Computed active columns. Total active columns: {int(mask.sum())}")
        return mask

    def combine_input_fields(self, input_vector: InputComposite) -> np.ndarray:
        """Prepare / combine input fields into a single binary numpy array.

        Responsibilities moved out of `compute_active_columns` so they can be used
        independently by the new single-step `run` helper or external callers.

        Accepted forms:
          - 1D array-like of ints (already concatenated)
          - list / tuple of field arrays ⇒ concatenated
          - dict[str, field array] ⇒ concatenated (order preserved) and field
            metadata (`field_ranges`, `field_order`, `column_field_map` reset)

        Returns: np.ndarray (dtype=int) of length == `input_space_size`.
        Raises: ValueError if combined length mismatches configured size.
        """
        # Dict branch (ordered by insertion order)
        if isinstance(input_vector, dict):
            start = 0
            arrays = []
            self.field_ranges = {}
            self.field_order = []
            for name, arr in input_vector.items():
                a = np.asarray(arr, dtype=int)
                end = start + a.shape[0]
                self.field_ranges[name] = (start, end)
                self.field_order.append(name)
                arrays.append(a)
                start = end
            combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
            # Invalidate previous mapping; will be recomputed lazily
            self.column_field_map = {}
        elif isinstance(input_vector, (list, tuple)):
            arrays = [np.asarray(v, dtype=int) for v in input_vector]
            combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
        else:
            combined = np.asarray(input_vector, dtype=int)

        if combined.shape[0] != self.input_space_size:
            raise ValueError(
                f"Combined input length {combined.shape[0]} != configured input_space_size {self.input_space_size}."
            )

        # Defensive structure check for previously stored metadata
        if self.field_ranges and any(len(rg) != 2 for rg in self.field_ranges.values()):
            self.field_ranges = {}
            self.field_order = []
            self.column_field_map = {}

        # Build mapping if new metadata present
        if self.field_ranges and not self.column_field_map:
            self._assign_column_fields()
        return combined

    # Backwards-compatible alias (private name used earlier in code/tests if any user code depended on it)
    _combine_input_fields = combine_input_fields

    def _columns_from_raw_input(self, combined: np.ndarray) -> List[Column]:
        """Return columns that receive at least one active (1) bit via a connected synapse.
        Used only in direct mode when an InputComposite is supplied and we want
        a non-competitive feed-forward activation set."""
        cols: List[Column] = []
        active_indices = np.where(combined > 0)[0]
        active_set = set(int(i) for i in active_indices)
        for col in self.columns:
            if any(s.source_input in active_set for s in col.connected_synapses):
                cols.append(col)
        return cols

    def _assign_column_fields(self) -> None:
        """Assign each column a dominant field based on connected synapse source indices.
        Dominant field = field with largest count of connected synapse inputs.
        Ties broken by earliest field order. Stores mapping in self.column_field_map."""
        if not self.field_ranges:
            return
        inv_order = {name: i for i, name in enumerate(self.field_order)}
        for col in self.columns:
            counts = {}
            for syn in col.connected_synapses:
                idx = syn.source_input
                for name, (s, e) in self.field_ranges.items():
                    if s <= idx < e:
                        counts[name] = counts.get(name, 0) + 1
                        break
            if counts:
                # select max count, break ties by field order
                best = sorted(counts.items(), key=lambda kv: (-kv[1], inv_order[kv[0]]))[0][0]
                self.column_field_map[col] = best
            else:
                self.column_field_map[col] = None

    # --------------------- Binary Conversion Utilities ---------------------
    def columns_to_binary(self, columns: Sequence[Column]) -> np.ndarray:
        """Return binary vector (len = number of columns) with 1 for each column in provided sequence/set."""
        mask = np.zeros(len(self.columns), dtype=int)
        col_index = {c: i for i, c in enumerate(self.columns)}
        for c in columns:
            idx = col_index.get(c)
            if idx is not None:
                mask[idx] = 1
        return mask

    def cells_to_binary(self, cells: Set[Cell]) -> np.ndarray:
        """Return binary vector over all cells (flattened columns) representing active/predictive/etc cells.

        Ordering = for col index i, its cells occupy slice [i*cells_per_column : (i+1)*cells_per_column)."""
        total_cells = len(self.columns) * self.cells_per_column
        vec = np.zeros(total_cells, dtype=int)
        # Build mapping once (could cache if performance concern)
        for col_idx, col in enumerate(self.columns):
            base = col_idx * self.cells_per_column
            for local_idx, cell in enumerate(col.cells):
                if cell in cells:
                    vec[base + local_idx] = 1
        return vec
    
    # --------------------- Temporal Memory Core ---------------------
    def compute_active_state(self, active_columns: Sequence[Column]) -> None:
        """Compute active & winner cells at time t using predictive cells from t-1.
        If a column was predicted (one or more predictive cells at t-1) only those predictive cells become active.
        Otherwise the column bursts (all cells active) and we pick a learning cell (winner cell)."""
        t = self.current_t
        prev_predictive = self.predictive_cells.get(t-1, set())
        active_cells_t = set()
        winner_cells_t = set()
        learning_segments_t = set()

        for column in active_columns:
            # Predictive cells from previous time in this column
            predictive_cells_prev = [cell for cell in column.cells if cell in prev_predictive]
            if predictive_cells_prev:
                # Column correctly predicted
                for cell in predictive_cells_prev:
                    active_cells_t.add(cell)
                    winner_cells_t.add(cell)
                    # Mark segments that were active at t-1 for learning
                    for seg in self._active_segments_of(cell, t-1):
                        learning_segments_t.add(seg)
            else:
                # Bursting: all cells active
                for cell in column.cells:
                    active_cells_t.add(cell)
                # Choose learning (winner) cell
                best_cell, best_segment = self.best_matching_cell(column, t-1)
                if best_segment is None:
                    # Create a new segment (ensure best_cell not None)
                    if best_cell is None:
                        # Fallback: choose first cell
                        best_cell = column.cells[0]
                    best_segment = Segment()
                    best_cell.segments.append(best_segment)
                winner_cells_t.add(best_cell)
                learning_segments_t.add(best_segment)

        self.active_cells[t] = active_cells_t
        self.winner_cells[t] = winner_cells_t
        self.learning_segments[t] = learning_segments_t
        print(f"Active state computed for time step {t}: {len(active_cells_t)} cells active.")

    def compute_predictive_state(self) -> None:
        """Compute predictive cells for next time based on segments active at time t.
        A segment is active if it has enough active connected synapses whose source cells are active at t."""
        t = self.current_t
        active_cells_t = self.active_cells.get(t, set())
        predictive_cells_t = set()
        for column in self.columns:
            for cell in column.cells:
                for seg in cell.segments:
                    if len(seg.active_synapses(active_cells_t)) >= SEGMENT_ACTIVATION_THRESHOLD:
                        predictive_cells_t.add(cell)
                        break
        self.predictive_cells[t] = predictive_cells_t
        print(f"Predictive state computed for time step {t}: {len(predictive_cells_t)} cells predictive.")

    def get_predictive_columns(
        self,
        t: Optional[int] = None,
        field_name: Optional[str] = None,
    ) -> np.ndarray:
        """Return binary vector of predictive columns (length = number of columns).

        Previous interface returned a set[Column]; now a numpy int array with 1s for
        predictive columns. Field filtering (if `field_name` supplied) is applied
        before vectorisation.
        """
        if not self.predictive_cells:
            return np.zeros(len(self.columns), dtype=int)
        max_t = max(self.predictive_cells.keys())
        if t is None:
            query_t = max_t
        elif t == -1:
            query_t = max_t - 1
        else:
            query_t = t
        if query_t < 0:
            return np.zeros(len(self.columns), dtype=int)
        pred_cells = self.predictive_cells.get(query_t, set())
        cols = {col for col in self.columns if any(cell in pred_cells for cell in col.cells)}
        if field_name is not None:
            # Allow either classic field_ranges-based metadata OR explicit column_field_map assignment
            if not (self.field_ranges or self.column_field_map):
                raise ValueError(
                    "Field-specific prediction requested but no field metadata available. "
                    "Use dict input of arrays (spatial) or dict of columns (direct) at least once before querying."
                )
            if self.field_ranges and not self.column_field_map:
                # Lazily derive mapping from proximal synapses if only field_ranges known
                self._assign_column_fields()
            cols = {c for c in cols if self.column_field_map.get(c) == field_name}
        return self.columns_to_binary(sorted(cols, key=lambda c: self.columns.index(c)))

    def reset_state(self) -> None:
        """Reset transient temporal memory state (cells' learned segments remain)."""
        self.active_cells = {}
        self.winner_cells = {}
        self.predictive_cells = {}
        self.learning_segments = {}
        self.negative_segments = {}

    def run(
        self,
    input_data: InputComposite | Sequence[Column] | Column | Dict[str, Union[Column, Sequence[Column]]],
        mode: str = "direct",
        inhibition_radius: Optional[float] = None,
    ) -> Dict[str, object]:
        """Execute one timestep of spatial + temporal processing.

        Parameters:
            input_data: Input composite or columns (mode dependent)
            mode: 'spatial' (default) or 'direct'
            inhibition_radius: required in spatial mode
        Returns:
            Dict with active_columns, active_cells, predictive_cells, learning_segments for current timestep.
        """
        t = self.current_t
        if mode not in {"spatial", "direct"}:
            raise ValueError("mode must be 'spatial' or 'direct'")
        if mode == "spatial":
            if inhibition_radius is None:
                raise ValueError("inhibition_radius required for spatial mode")
            active_mask = self.compute_active_columns(input_data, inhibition_radius)  # type: ignore[arg-type]
            active_columns = [self.columns[i] for i, v in enumerate(active_mask) if v]
        else:
            active_columns = self._direct_mode_active_columns(input_data)
            active_mask = self.columns_to_binary(active_columns)

        active_columns = [c for c in active_columns if isinstance(c, Column)]
        self.compute_active_state(active_columns)
        self.compute_predictive_state()
        self.learn()
        self.current_t += 1
        # Vectorised cell-related outputs
        active_cells_vec = self.cells_to_binary(self.active_cells.get(t, set()))
        predictive_cells_vec = self.cells_to_binary(self.predictive_cells.get(t, set()))
        # Learning segments -> mark owning cells (winner cells) as learning
        learning_cells_vec = self.cells_to_binary(self.winner_cells.get(t, set()))
        return {
            "active_columns": active_mask,
            "active_cells": active_cells_vec,
            "predictive_cells": predictive_cells_vec,
            "learning_cells": learning_cells_vec,
        }

    def _direct_mode_active_columns(
        self,
        input_data: InputComposite | Sequence[Column] | Column | Dict[str, Union[Column, Sequence[Column]]],
    ) -> List[Column]:
        """Resolve active columns for 'direct' mode without spatial competition (duplicate of htm.py)."""
        if isinstance(input_data, Column):
            return [input_data]
        if isinstance(input_data, dict):
            values = list(input_data.values())
            is_grouping = bool(values) and all(isinstance(v, (Column, list, tuple)) for v in values)
            if is_grouping:
                active: List[Column] = []
                self.field_order = list(input_data.keys())
                self.field_ranges = {}
                for fname, spec in input_data.items():
                    if isinstance(spec, Column):
                        iter_items = [spec]
                    else:
                        iter_items = list(spec)  # type: ignore[arg-type]
                    for item in iter_items:
                        if isinstance(item, int):
                            col_obj = self.columns[item]
                        else:
                            col_obj = item  # type: ignore[assignment]
                        if isinstance(col_obj, Column):
                            active.append(col_obj)
                            self.column_field_map[col_obj] = fname
                seen: Set[Column] = set()
                deduped: List[Column] = []
                for c in active:
                    if c not in seen:
                        deduped.append(c)
                        seen.add(c)
                return deduped
            combined = self.combine_input_fields(input_data)  # type: ignore[arg-type]
            return self._columns_from_raw_input(combined)
        if isinstance(input_data, np.ndarray):
            combined = self.combine_input_fields(input_data)
            return self._columns_from_raw_input(combined)
        if isinstance(input_data, (list, tuple)):
            seq = list(input_data)
            if not seq:
                return []
            first = seq[0]
            if isinstance(first, Column):
                return seq  # type: ignore[return-value]
            if isinstance(first, (int, np.integer)):
                return [self.columns[int(i)] for i in seq if isinstance(i, (int, np.integer))]
            combined = self.combine_input_fields(seq)  # type: ignore[arg-type]
            return self._columns_from_raw_input(combined)
        combined = self.combine_input_fields(input_data)  # type: ignore[arg-type]
        return self._columns_from_raw_input(combined)

    def learn(self) -> None:
        """Apply learning updates after computing active and predictive states at time t.
        - Reinforce learning segments for winner cells (positive).
        - Punish segments that predicted (at t-1) but whose columns did not become active (negative)."""
        t = self.current_t  # internal time
        prev_predictive = self.predictive_cells.get(t-1, set())
        active_columns = {c for c in self.columns if any(cell in self.active_cells.get(t, set()) for cell in c.cells)}
        # Negative segments: segments that were active at t-1 (causing prediction) but column not active at t
        negative_segments = set()
        for column in self.columns:
            if column not in active_columns:
                for cell in column.cells:
                    if cell in prev_predictive:
                        for seg in self._active_segments_of(cell, t-1):
                            negative_segments.add(seg)
        self.negative_segments[t] = negative_segments

        # Positive reinforcement
        for seg in self.learning_segments.get(t, set()):
            self.reinforce_segment(seg)
        # Negative reinforcement
        for seg in negative_segments:
            self.punish_segment(seg)
        print(f"Learning applied at time {t}: +{len(self.learning_segments.get(t, set()))} / -{len(negative_segments)} segments.")

    # --------------------- Helper Methods (TM) ---------------------
    def best_matching_cell(self, column: Column, prev_t: int) -> Tuple[Optional[Cell], Optional[Segment]]:
        prev_active_cells = self.active_cells.get(prev_t, set())
        best_cell = None
        best_segment = None
        best_match = -1
        for cell in column.cells:
            if not cell.segments:
                # Prefer an unused cell immediately if no better match found yet
                if best_match == -1:
                    best_cell = cell
                    best_segment = None
                    best_match = 0
                continue
            for seg in cell.segments:
                match_count = len(seg.matching_synapses(prev_active_cells))
                if match_count > best_match and match_count >= 0:
                    best_match = match_count
                    best_cell = cell
                    best_segment = seg
        return best_cell, best_segment

    def _active_segments_of(self, cell: Cell, t: int) -> List[Segment]:
        prev_active_cells = self.active_cells.get(t, set())
        active_list = []
        for seg in cell.segments:
            if len(seg.active_synapses(prev_active_cells)) >= SEGMENT_ACTIVATION_THRESHOLD:
                active_list.append(seg)
        return active_list
    
    def active_segments_of(self, cell: Cell) -> List[Segment]:
        """Active segments for this cell at current timestep (uses self.current_t)."""
        return self._active_segments_of(cell, self.current_t)

    def reinforce_segment(self, segment: Segment) -> None:
        t = self.current_t  # internal time
        prev_active_cells = self.active_cells.get(t-1, set())
        # Strengthen existing active synapses
        for syn in segment.synapses:
            if syn.source_cell in prev_active_cells:
                syn.permanence = min(1.0, syn.permanence + PERMANENCE_INC)
            else:
                syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)
        # Grow new synapses (sample from prev active cells not already connected)
        existing_sources = {syn.source_cell for syn in segment.synapses}
        candidates = [c for c in prev_active_cells if c not in existing_sources]
        random.shuffle(candidates)
        for cell_src in candidates[:NEW_SYNAPSE_MAX]:
            segment.synapses.append(DistalSynapse(cell_src, INITIAL_DISTAL_PERM))
        segment.sequence_segment = True

    def punish_segment(self, segment: Segment) -> None:
        for syn in segment.synapses:
            syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)



    # Applies inhibition to determine which columns will become active
    def inhibition(self, columns: Sequence[Column], inhibition_radius: float) -> List[Column]:
        active_columns: List[Column] = []
        for c in columns:
        # Find neighbors of column c
            neighbors = [c2 for c2 in columns if c != c2 and self.euclidean_distance(c.position, c2.position) <= inhibition_radius]
            min_local_activity = self.kth_score(neighbors, DESIRED_LOCAL_ACTIVITY)
            if c.overlap > 0 and c.overlap >= min_local_activity:
                active_columns.append(c)
        print(f"After inhibition, active columns: {[c.position for c in active_columns]}")
        return active_columns

    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate the Euclidean distance between two points."""
        return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))


    # Returns the k-th highest overlap value from a list of columns
    def kth_score(self, neighbors: Sequence[Column], k: int) -> float:
        if not neighbors:
            return 0
        ordered = sorted(neighbors, key=lambda x: x.overlap, reverse=True)
        if k <= 0:
            return 0
        if k > len(ordered):
            # If fewer neighbors than desired local activity, use lowest neighbor overlap
            return float(ordered[-1].overlap) if ordered else 0.0
        return float(ordered[k-1].overlap)


    # Removed legacy helper methods tied to old temporal representation.


    
    def learning_phase(self, active_columns: Sequence[Column], input_vector: np.ndarray) -> None:
        """Spatial Pooler permanence adaptation for currently active columns."""
        for c in active_columns:
            for s in c.potential_synapses:
                if input_vector[s.source_input]:
                    s.permanence = min(1.0, s.permanence + PERMANENCE_INC)
                else:
                    s.permanence = max(0.0, s.permanence - PERMANENCE_DEC)
            c.connected_synapses = [s for s in c.potential_synapses if s.permanence > CONNECTED_PERM]
        print(f"Learning phase updated the synapses for {len(active_columns)} active columns.")

        # Could update inhibition radius if implementing adaptive inhibition.
        _ = self.average_receptive_field_size(self.columns)

    def average_receptive_field_size(self, columns: Sequence[Column]) -> float:
        total_receptive_field_size = 0
        count = 0
        for c in columns:
            connected_positions = [s.source_input for s in c.connected_synapses]
            if connected_positions:
                receptive_field_size = max(connected_positions) - min(connected_positions)
                total_receptive_field_size += receptive_field_size
                count += 1
        return total_receptive_field_size / count if count > 0 else 0.0
    
if __name__ == "__main__":  # Optional manual smoke usage placeholder
    input_space_size = 100
    column_count = 256  # reduced for demo runtime
    cells_per_column = 8
    initial_synapses_per_column = 20
    steps = 5
    inhibition_radius = 10

    tp = TemporalPooler(input_space_size, column_count, cells_per_column, initial_synapses_per_column)

    for t in range(steps):
        tp.current_t = t
        input_vector = np.random.randint(2, size=input_space_size)
        active_mask = tp.compute_active_columns(input_vector, inhibition_radius)
        active_list = [tp.columns[i] for i, v in enumerate(active_mask) if v]
        tp.compute_active_state(active_list)
        tp.compute_predictive_state()
        tp.learn()
        if t == 0:
            # Spatial pooler learning only occasionally (for demo call once)
            tp.learning_phase(active_list, input_vector)
    print("Temporal Memory demo completed.")
    tp = TemporalPooler(input_space_size, column_count, cells_per_column, initial_synapses_per_column)

    for t in range(steps):
        tp.current_t = t
        input_vector = np.random.randint(2, size=input_space_size)
        active_mask = tp.compute_active_columns(input_vector, inhibition_radius)
        active_list = [tp.columns[i] for i, v in enumerate(active_mask) if v]
        tp.compute_active_state(active_list)
        tp.compute_predictive_state()
        tp.learn()
        if t == 0:
            # Spatial pooler learning only occasionally (for demo call once)
            tp.learning_phase(active_list, input_vector)
    print("Temporal Memory demo completed.")