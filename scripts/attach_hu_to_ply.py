#!/usr/bin/env python3
"""Sample CT intensities at Gaussian centers and append them to a PLY file.

The Gaussian positions are interpreted as physical/world coordinates.  The CT
origin, spacing, and direction matrix are therefore all used when converting a
position to a continuous voxel index.  Intensities are sampled with trilinear
interpolation and stored as a new ``property float hu`` on every PLY vertex.

The input PLY is never modified.  This script currently accepts binary PLY
files whose first data element is ``vertex`` and whose vertex properties are
all scalar values, which matches the point clouds written by the training
repository.
"""

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk


PLY_SCALAR_TYPES: Dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True)
class PlyHeader:
    lines: List[bytes]
    data_offset: int
    vertex_count: int
    vertex_dtype: np.dtype
    byte_order: str
    hu_insert_index: int
    has_hu: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a CT volume at every Gaussian center and append a float HU "
            "property to a binary PLY point cloud."
        )
    )
    parser.add_argument("--ct", required=True, type=Path, help="Input CT NIfTI image.")
    parser.add_argument(
        "--input-ply", required=True, type=Path, help="Gaussian PLY without HU."
    )
    parser.add_argument(
        "--output-ply", required=True, type=Path, help="New Gaussian PLY with HU."
    )
    parser.add_argument(
        "--outside",
        choices=("error", "clamp", "nan"),
        default="error",
        help=(
            "How to handle Gaussian centers outside the CT physical extent "
            "(default: error)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of Gaussian centers sampled per chunk (default: 1000000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file. The input is still preserved.",
    )
    return parser.parse_args()


def read_ply_header(path: Path) -> PlyHeader:
    lines: List[bytes] = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header ended before end_header")
            lines.append(line)
            if line.strip() == b"end_header":
                break
        data_offset = stream.tell()

    decoded = [line.decode("ascii").strip() for line in lines]
    if not decoded or decoded[0] != "ply":
        raise ValueError("Input is not a PLY file")

    format_tokens = next(
        (line.split() for line in decoded if line.startswith("format ")), None
    )
    if format_tokens is None or len(format_tokens) < 2:
        raise ValueError("PLY header has no valid format declaration")
    ply_format = format_tokens[1]
    if ply_format == "binary_little_endian":
        byte_order = "<"
    elif ply_format == "binary_big_endian":
        byte_order = ">"
    else:
        raise ValueError(
            f"Unsupported PLY format '{ply_format}'; a binary PLY is required"
        )

    elements: List[Tuple[str, int]] = []
    current_element = None
    vertex_count = None
    vertex_fields: List[Tuple[str, str]] = []
    vertex_property_line_indices: List[int] = []

    for line_index, line in enumerate(decoded):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "element":
            if len(tokens) != 3:
                raise ValueError(f"Invalid element declaration: {line}")
            current_element = tokens[1]
            count = int(tokens[2])
            elements.append((current_element, count))
            if current_element == "vertex":
                vertex_count = count
        elif tokens[0] == "property" and current_element == "vertex":
            if len(tokens) != 3:
                raise ValueError("List-valued vertex properties are not supported")
            ply_type, name = tokens[1], tokens[2]
            if ply_type not in PLY_SCALAR_TYPES:
                raise ValueError(f"Unsupported PLY scalar type '{ply_type}'")
            vertex_fields.append((name, byte_order + PLY_SCALAR_TYPES[ply_type]))
            vertex_property_line_indices.append(line_index)

    if vertex_count is None:
        raise ValueError("PLY has no vertex element")
    if not elements or elements[0][0] != "vertex":
        raise ValueError("The vertex element must be the first PLY data element")
    if not vertex_fields:
        raise ValueError("PLY vertex element has no scalar properties")
    for required_name in ("x", "y", "z"):
        if required_name not in {name for name, _ in vertex_fields}:
            raise ValueError(f"PLY vertex is missing required property '{required_name}'")

    return PlyHeader(
        lines=lines,
        data_offset=data_offset,
        vertex_count=vertex_count,
        vertex_dtype=np.dtype(vertex_fields, align=False),
        byte_order=byte_order,
        hu_insert_index=max(vertex_property_line_indices) + 1,
        has_hu="hu" in {name for name, _ in vertex_fields},
    )


def load_vertices(path: Path, header: PlyHeader) -> np.ndarray:
    with path.open("rb") as stream:
        stream.seek(header.data_offset)
        vertices = np.fromfile(
            stream, dtype=header.vertex_dtype, count=header.vertex_count
        )
    if len(vertices) != header.vertex_count:
        raise ValueError(
            f"PLY contains {len(vertices)} complete vertices, expected "
            f"{header.vertex_count}"
        )
    return vertices


def physical_points_to_continuous_indices(
    points_xyz: np.ndarray, image: sitk.Image
) -> np.ndarray:
    """Vectorized SimpleITK physical-point to continuous-index transform."""
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    inverse_direction = np.linalg.inv(direction)
    return ((points_xyz - origin) @ inverse_direction.T) / spacing


def trilinear_sample(
    volume_zyx: np.ndarray,
    indices_xyz: np.ndarray,
    outside_mode: str,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    size_xyz = np.asarray(volume_zyx.shape[::-1], dtype=np.int64)
    lower_ok = np.all(indices_xyz >= 0.0, axis=1)
    upper_ok = np.all(indices_xyz <= (size_xyz - 1), axis=1)
    valid = lower_ok & upper_ok

    invalid_count = int((~valid).sum())
    if invalid_count and outside_mode == "error":
        first = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            f"{invalid_count} Gaussian center(s) are outside the CT; first invalid "
            f"continuous index is {indices_xyz[first].tolist()}. Use --outside clamp "
            "or --outside nan only if this mismatch is intentional."
        )

    sampled = np.full(len(indices_xyz), np.nan, dtype=np.float32)
    sample_indices = np.clip(indices_xyz, 0.0, size_xyz - 1)

    for start in range(0, len(indices_xyz), chunk_size):
        end = min(start + chunk_size, len(indices_xyz))
        if outside_mode == "nan":
            selected = np.flatnonzero(valid[start:end]) + start
        else:
            selected = np.arange(start, end)
        if len(selected) == 0:
            continue

        coordinates = sample_indices[selected]
        base = np.floor(coordinates).astype(np.int64)
        upper = np.minimum(base + 1, size_xyz - 1)
        fraction = coordinates - base

        x0, y0, z0 = base.T
        x1, y1, z1 = upper.T
        wx, wy, wz = fraction.T

        c000 = volume_zyx[z0, y0, x0]
        c100 = volume_zyx[z0, y0, x1]
        c010 = volume_zyx[z0, y1, x0]
        c110 = volume_zyx[z0, y1, x1]
        c001 = volume_zyx[z1, y0, x0]
        c101 = volume_zyx[z1, y0, x1]
        c011 = volume_zyx[z1, y1, x0]
        c111 = volume_zyx[z1, y1, x1]

        c00 = c000 * (1.0 - wx) + c100 * wx
        c10 = c010 * (1.0 - wx) + c110 * wx
        c01 = c001 * (1.0 - wx) + c101 * wx
        c11 = c011 * (1.0 - wx) + c111 * wx
        c0 = c00 * (1.0 - wy) + c10 * wy
        c1 = c01 * (1.0 - wy) + c11 * wy
        sampled[selected] = (c0 * (1.0 - wz) + c1 * wz).astype(np.float32)

    return sampled, valid


def output_header_bytes(header: PlyHeader) -> bytes:
    lines = list(header.lines)
    newline = b"\r\n" if lines[0].endswith(b"\r\n") else b"\n"
    lines.insert(header.hu_insert_index, b"property float hu" + newline)
    return b"".join(lines)


def write_augmented_ply(
    input_path: Path,
    output_path: Path,
    header: PlyHeader,
    vertices: np.ndarray,
    hu: np.ndarray,
    chunk_size: int,
) -> None:
    hu_dtype = header.byte_order + "f4"
    output_dtype = np.dtype(header.vertex_dtype.descr + [("hu", hu_dtype)])
    input_vertex_bytes = header.vertex_count * header.vertex_dtype.itemsize
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as destination:
            destination.write(output_header_bytes(header))
            for start in range(0, header.vertex_count, chunk_size):
                end = min(start + chunk_size, header.vertex_count)
                output_vertices = np.empty(end - start, dtype=output_dtype)
                for name in header.vertex_dtype.names or ():
                    output_vertices[name] = vertices[name][start:end]
                output_vertices["hu"] = hu[start:end]
                output_vertices.tofile(destination)

            with input_path.open("rb") as source:
                source.seek(header.data_offset + input_vertex_bytes)
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.replace(str(temp_path), str(output_path))
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def format_distribution(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return "no finite samples"
    p05, p25, p50, p75, p95 = np.percentile(finite, [5, 25, 50, 75, 95])
    return (
        f"count={len(finite):7d}  p05={p05:8.1f}  p25={p25:8.1f}  "
        f"median={p50:8.1f}  p75={p75:8.1f}  p95={p95:8.1f}"
    )


def print_summary(vertices: np.ndarray, hu: np.ndarray, valid: np.ndarray) -> None:
    print(f"CT coverage: {int(valid.sum())}/{len(valid)} ({valid.mean() * 100:.3f}%)")
    print(f"All layers: {format_distribution(hu)}")
    if "layer" not in (vertices.dtype.names or ()):
        return
    layers = vertices["layer"]
    for layer in np.unique(layers):
        layer_values = hu[layers == layer]
        print(f"Layer {int(layer):2d}: {format_distribution(layer_values)}")


def validate_paths(args: argparse.Namespace) -> None:
    if not args.ct.is_file():
        raise FileNotFoundError(f"CT image not found: {args.ct}")
    if not args.input_ply.is_file():
        raise FileNotFoundError(f"Input PLY not found: {args.input_ply}")
    if args.input_ply.resolve() == args.output_ply.resolve():
        raise ValueError("Input and output PLY paths must be different")
    if args.output_ply.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_ply} (pass --overwrite to replace it)"
        )
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")


def main() -> int:
    args = parse_args()
    try:
        validate_paths(args)
        print(f"Reading PLY: {args.input_ply}")
        header = read_ply_header(args.input_ply)
        if header.has_hu:
            raise ValueError("Input PLY already contains a 'hu' property")
        vertices = load_vertices(args.input_ply, header)

        print(f"Reading CT:  {args.ct}")
        image = sitk.ReadImage(str(args.ct))
        if image.GetDimension() != 3:
            raise ValueError(f"Expected a 3D CT image, got {image.GetDimension()}D")
        volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)

        points = np.column_stack(
            (vertices["x"], vertices["y"], vertices["z"])
        ).astype(np.float64, copy=False)
        indices = physical_points_to_continuous_indices(points, image)
        hu, valid = trilinear_sample(
            volume, indices, args.outside, args.chunk_size
        )
        print_summary(vertices, hu, valid)

        print(f"Writing:     {args.output_ply}")
        write_augmented_ply(
            args.input_ply,
            args.output_ply,
            header,
            vertices,
            hu,
            args.chunk_size,
        )
        print("Done. Added 'property float hu' to every Gaussian vertex.")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
