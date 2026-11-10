"""Smooth small v4 mesh corrugations with bounded physical displacement."""
import argparse
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def volume(mesh):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    return mass.GetVolume() / 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-displacement-mm', type=float, default=0.6)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--pass-band', type=float, default=0.03)
    args = parser.parse_args()
    if args.output.exists() or args.max_displacement_mm <= 0:
        raise ValueError('Use a new output directory and positive displacement limit')
    reader = vtk.vtkPLYReader()
    reader.SetFileName(str(args.input))
    reader.Update()
    mesh = reader.GetOutput()
    original = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh)
    smoother.SetNumberOfIterations(args.iterations)
    smoother.SetPassBand(args.pass_band)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOff()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(smoother.GetOutput())
    delta = vtk_to_numpy(result.GetPoints().GetData()).astype(np.float64) - original
    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, args.max_displacement_mm / np.maximum(lengths, 1e-12))[:, None]
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk((original + delta).astype(np.float32), deep=True))
    result.SetPoints(points)
    before, after = volume(mesh), volume(result)
    if abs(after / before - 1) > 0.05:
        raise ValueError(f'Mesh volume changed more than 5%: {before} -> {after} mL')
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(result)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    if edges.GetOutput().GetNumberOfCells():
        raise ValueError('Smoothed surface has boundary/nonmanifold edges')
    args.output.mkdir(parents=True)
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(str(args.output / 'L7_surface_smoothed.ply'))
    writer.SetInputData(result)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError('Surface export failed')
    report = {
        'source_mesh': str(args.input), 'points': len(original),
        'iterations': args.iterations, 'pass_band': args.pass_band,
        'max_displacement_mm': args.max_displacement_mm,
        'actual_displacement_p50_p95_max_mm': np.percentile(np.linalg.norm(delta, axis=1), [50, 95, 100]).tolist(),
        'volume_before_ml': before, 'volume_after_ml': after,
        'volume_change_percent': (after / before - 1) * 100,
        'open_or_nonmanifold_edges': 0,
        'scope': 'Surface smoothing derived from corrected v4; native segmentation mask is unchanged. Surface CT-HU compliance must not be copied from v4 metrics.'
    }
    (args.output / 'surface_smoothing.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
