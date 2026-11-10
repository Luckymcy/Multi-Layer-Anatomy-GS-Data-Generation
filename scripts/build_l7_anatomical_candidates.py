"""CT-supported L7 range revision, with explicit uncertainty without pericardium.

Uses cardiac chambers to bound the heart body, restores groove/internal fat,
and grows only through CT fat. Outputs candidate masks and review images;
these are not validated epicardial-fat labels in the absence of a pericardium.
No existing masks, teachers or trained Gaussian models are overwritten.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import generate_heart_exterior_teachers_v4 as exterior
import generate_heart_teachers_v2 as legacy
import generate_heart_teachers_v5_sdf as surfaces
import attach_hu_to_ply as hu_reader


def geometry(image):
    return image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()


def image_from(array, reference):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def distance(mask, reference):
    if not mask.any():
        return np.full(mask.shape, 1000, dtype=np.float32)
    image = sitk.SignedMaurerDistanceMap(image_from(mask.astype(np.uint8), reference),
        insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    return sitk.GetArrayFromImage(image)


def connected_to_seeds(candidate, seeds, reference, minimum_ml=0.02):
    labels_img = sitk.ConnectedComponent(image_from(candidate.astype(np.uint8), reference), False)
    labels = sitk.GetArrayFromImage(labels_img)
    counts = np.bincount(labels.ravel())
    selected = np.unique(labels[seeds & candidate])
    min_count = int(np.ceil(minimum_ml * 1000 / np.prod(reference.GetSpacing())))
    selected = selected[(selected != 0) & (counts[selected] >= min_count)]
    return np.isin(labels, selected)


def erode2d(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return mask & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]


def render_review(path, actors, pose, center, radius):
    """Use lower fill lights as well as upper lights to make range inspection fair."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0,0,0)
    renderer.SetBackgroundAlpha(0)
    exterior.add_lights(renderer,center,radius)
    for direction in ((0,-0.8,-1.5),(-0.7,0.9,-0.8)):
        light=vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(*(center+np.asarray(direction)*radius*2.2))
        light.SetFocalPoint(*center)
        light.SetColor(1,1,1)
        light.SetIntensity(0.65)
        renderer.AddLight(light)
    window=vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(8)
    window.SetSize(1024,1024)
    window.AddRenderer(renderer)
    for actor in actors:
        renderer.AddActor(actor)
    legacy.configure_camera(renderer.GetActiveCamera(),pose[0],pose[1],60)
    renderer.ResetCameraClippingRange()
    window.Render()
    capture=vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGBA()
    capture.ReadFrontBufferOff()
    capture.Update()
    pixels=vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(1024,1024,4)[::-1]
    Image.fromarray(pixels).resize((768,768),Image.Resampling.LANCZOS).save(path)
    renderer.RemoveAllViewProps()
    window.Finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pericardium-mask', type=Path,
                        help='Optional native CT-grid mask of the interior of the pericardial sac')
    parser.add_argument('--hu-low', type=float, default=-190)
    parser.add_argument('--hu-high', type=float, default=-30)
    parser.add_argument('--surface-sigma-mm', type=float, default=0.35)
    parser.add_argument('--occupancy-threshold', type=float, default=0.45)
    args = parser.parse_args()
    if args.hu_low >= args.hu_high-10:
        parser.error('HU range must leave room for the 5 HU surface margin at both ends')
    if args.surface_sigma_mm <= 0 or not 0 < args.occupancy_threshold < 1:
        parser.error('Surface sigma must be positive and occupancy threshold must be between 0 and 1')
    root, output = args.root.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a nonempty review directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    ct_img = sitk.ReadImage(str(root / 'data/ct/nifti/DSCorAdSeq075B26fBestDiast75_7.nii.gz'))
    ct = sitk.GetArrayFromImage(ct_img)

    def read_mask(relative):
        image = sitk.ReadImage(str(root / relative))
        if geometry(image) != geometry(ct_img):
            raise ValueError(f'CT grid mismatch: {relative}')
        return sitk.GetArrayFromImage(image) > 0

    heart = read_mask('data/ct/masks/total_free/heart.nii.gz')
    chamber_names = ['heart_ventricle_left', 'heart_ventricle_right',
                     'heart_atrium_left', 'heart_atrium_right']
    full_chambers = [read_mask(f'data/ct/masks/heartchambers/{name}.nii.gz') for name in chamber_names]
    myocardium = read_mask('data/ct/masks/heartchambers/heart_myocardium.nii.gz')
    body = np.logical_or.reduce(full_chambers + [myocardium])
    coronary = read_mask('data/ct/masks/layers/L6_coronary_arteries.nii.gz')
    body_z = np.where(body)[0]
    spacing = np.asarray(ct_img.GetSpacing()[::-1])
    # Cardiac-body limits exclude remote fat around the long great-vessel stumps.
    zmin = max(0, int(body_z.min() - np.ceil(1 / spacing[0])))
    zmax = min(ct.shape[0], int(body_z.max() + np.ceil(6 / spacing[0]) + 1))
    coords = np.asarray(np.where(heart))
    pad = np.ceil(12 / spacing).astype(int)
    lo = np.maximum(coords.min(axis=1) - pad, 0)
    hi = np.minimum(coords.max(axis=1) + pad + 1, ct.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    reference = exterior.crop_roi(ct_img, lo, hi)
    hu, h, b, cor = ct[box], heart[box], body[box], coronary[box]
    chambers = [m[box] for m in full_chambers]
    d_heart, d_body, d_cor = distance(h, reference), distance(b, reference), distance(cor, reference)
    d_chambers = [distance(m, reference) for m in chambers]
    d_vent = np.minimum(d_chambers[0], d_chambers[1])
    d_atria = np.minimum(d_chambers[2], d_chambers[3])
    # These are spatial hints, not a claimed segmentation of the grooves.
    ventricular_groove = (d_chambers[0] <= 7) & (d_chambers[1] <= 7)
    av_groove = (d_vent <= 8) & (d_atria <= 8)
    hints = ventricular_groove | av_groove | (d_cor <= 6)
    protected = np.logical_or.reduce([d < -0.8 for d in d_chambers])
    protected |= distance(myocardium[box], reference) < -0.8
    z = np.arange(lo[0], hi[0])[:, None, None]
    axial_limits = (z >= zmin) & (z < zmax)
    fat_hu = (hu >= args.hu_low) & (hu <= args.hu_high)
    base_proxy = fat_hu & ~h & (d_heart <= 4)
    # HU continuity is the tissue barrier. Distances below are broad safety
    # envelopes, not a required fat thickness. Keep broader extensions separate.
    if args.pericardium_mask:
        pericardium = read_mask(args.pericardium_mask)[box]
        envelope = pericardium & axial_limits
        broad_envelope = envelope
    else:
        envelope = axial_limits & (d_body <= 12) & ((d_heart <= 5.5) | (hints & (d_heart <= 8)))
        broad_envelope = axial_limits & (d_body <= 12) & (d_heart <= 10)
    internal = fat_hu & h & ~protected & (d_body <= 8) & axial_limits
    seeds = fat_hu & (hu <= -50) & ~protected & axial_limits & (
        (d_heart <= 1.5) | internal | (hints & (d_cor <= 3)))
    admissible = fat_hu & ~protected & envelope
    candidate = connected_to_seeds(admissible, seeds, reference)
    if not candidate.any():
        raise ValueError('No fat candidate connected to cardiac seeds; review the CT and masks')
    broad = connected_to_seeds(fat_hu & ~protected & broad_envelope, seeds, reference)
    uncertain = broad & ~candidate
    masks = {
        'L7_corrected_candidate_ct_grid': candidate,
        'L7_internal_groove_additions_ct_grid': candidate & h,
        'L7_uncertain_outer_extension_ct_grid': uncertain,
        'old_4mm_ct_proxy': base_proxy,
        'added_vs_4mm_proxy': candidate & ~base_proxy,
        'removed_vs_4mm_proxy': base_proxy & ~candidate,
    }
    for name, local in masks.items():
        full = np.zeros(ct.shape, dtype=np.uint8)
        full[box] = local
        sitk.WriteImage(image_from(full, ct_img), str(output / f'{name}.nii.gz'), True)
    voxel_ml = float(np.prod(ct_img.GetSpacing()) / 1000)
    metrics = {
        'status': 'CT-supported anatomical candidate; pericardial classification unvalidated',
        'pericardium_mask': str(args.pericardium_mask) if args.pericardium_mask else None,
        'ct_spacing_xyz_mm': ct_img.GetSpacing(), 'hu_band': [args.hu_low, args.hu_high],
        'old_comparison': 'HU plus external 4mm rule reconstructed without missing torso_fat; not exact old mask',
        'cardiac_body_axial_limits_native_z': [zmin, zmax-1],
        'volume_ml': {name: float(mask.sum()*voxel_ml) for name, mask in masks.items()},
        'candidate_hu_p05_p50_p95': np.percentile(hu[candidate], [5,50,95]).tolist(),
        'candidate_hu_band_percent': float(np.mean(fat_hu[candidate])*100),
        'candidate_components': int(sitk.GetArrayFromImage(sitk.RelabelComponent(
            sitk.ConnectedComponent(image_from(candidate.astype(np.uint8),reference)))).max()),
        'surface_sigma_mm': args.surface_sigma_mm,
        'occupancy_threshold': args.occupancy_threshold,
        'volume_preservation_forced': False,
        'important_limit': 'No automatic rule here verifies the thin pericardium. Uncertain extensions are not merged into L7.',
    }
    print(json.dumps(metrics,indent=2),flush=True)

    # Continuous signed-distance surfaces, with small smoothing and explicit CT-edge caps.
    def make_surface(mask, sigma):
        image = image_from(mask.astype(np.uint8), reference)
        sdf = sitk.SignedMaurerDistanceMap(image, insideIsPositive=True,
                                          squaredDistance=False, useImageSpacing=True)
        sdf += min(image.GetSpacing())*0.5
        iso = exterior.resample_isotropic(sdf,0.4,sitk.sitkLinear,-20,sitk.sitkFloat32)
        if sigma:
            iso = sitk.SmoothingRecursiveGaussian(iso,sigma)
        # No remote displacement or enforced restoration of the old volume.
        iso = sitk.ConstantPad(iso,[1]*3,[1]*3,-20)
        return surfaces.surface_from_sdf(iso,12,0.15)

    def make_fat_surface(mask):
        # Smooth native occupancy before resampling. This retains thin sheets
        # better than blurring signed distances, without fitting to an old volume.
        image=image_from(mask.astype(np.uint8),reference)
        probability=sitk.SmoothingRecursiveGaussian(sitk.Cast(image,sitk.sitkFloat32),
                                                    args.surface_sigma_mm)
        probability=exterior.resample_isotropic(probability,0.4,sitk.sitkLinear,0,sitk.sitkFloat32)
        source_sdf=sitk.SignedMaurerDistanceMap(image,insideIsPositive=True,
            squaredDistance=False,useImageSpacing=True)+min(image.GetSpacing())*0.5
        source_sdf=sitk.Resample(source_sdf,probability,sitk.Transform(3,sitk.sitkIdentity),
                                 sitk.sitkLinear,-20,sitk.sitkFloat32)
        hu_iso=sitk.Resample(sitk.Cast(reference,sitk.sitkFloat32),probability,
            sitk.Transform(3,sitk.sitkIdentity),sitk.sitkLinear,-1000,sitk.sitkFloat32)
        surface_hu=sitk.GetArrayFromImage(hu_iso)
        # A binary-occupancy boundary can otherwise bulge into muscle/air after
        # smoothing. Constrain the continuous contour with actual CT intensity.
        # Five HU of inset provides a modest interpolation safety margin.
        hu_support=np.minimum((surface_hu-(args.hu_low+5))/160,
                              ((args.hu_high-5)-surface_hu)/160)
        field_array=np.minimum(sitk.GetArrayFromImage(probability)-args.occupancy_threshold,
                               sitk.GetArrayFromImage(source_sdf)+0.3).astype(np.float32)
        field_array=np.minimum(field_array,hu_support).astype(np.float32)
        field=sitk.ConstantPad(image_from(field_array,probability),[1]*3,[1]*3,-1)
        return surfaces.surface_from_sdf(field,12,0.15),field

    fat_surface,fat_field=make_fat_surface(candidate)
    sitk.WriteImage(fat_field,str(output/'L7_corrected_surface_field_iso.nii.gz'),True)
    field_native=sitk.Resample(sitk.Cast(fat_field>0,sitk.sitkUInt8),reference,
        sitk.Transform(3,sitk.sitkIdentity),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    smoothed_native=sitk.GetArrayFromImage(field_native)>0
    metrics['surface_source_voxels_retained_percent']=float(
        np.sum(smoothed_native&candidate)/candidate.sum()*100)
    metrics['surface_range_native_dice']=float(2*np.sum(smoothed_native&candidate)/
        (smoothed_native.sum()+candidate.sum()))
    metrics['surface_max_outward_support_mm']=0.3
    metrics['surface_ct_hu_support_band']=[args.hu_low+5,args.hu_high-5]
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(str(output/'L7_corrected_surface.ply'))
    writer.SetInputData(fat_surface)
    writer.SetFileTypeToBinary()
    writer.Write()
    edge = vtk.vtkFeatureEdges()
    edge.SetInputData(fat_surface)
    edge.BoundaryEdgesOn(); edge.NonManifoldEdgesOn(); edge.FeatureEdgesOff(); edge.ManifoldEdgesOff()
    edge.Update()
    metrics['mesh_open_or_nonmanifold_edges'] = edge.GetOutput().GetNumberOfCells()
    mass=vtk.vtkMassProperties()
    mass.SetInputData(fat_surface)
    mass.Update()
    metrics['smoothed_surface_volume_ml']=mass.GetVolume()/1000
    points=vtk_to_numpy(fat_surface.GetPoints().GetData()).astype(np.float64)
    point_indices=hu_reader.physical_points_to_continuous_indices(points,ct_img)
    sampled_hu,in_bounds=hu_reader.trilinear_sample(ct,point_indices,'nan',100000)
    metrics['mesh_vertex_hu_p05_p50_p95']=np.percentile(sampled_hu[in_bounds],[5,50,95]).tolist()
    metrics['mesh_vertices_in_fat_hu_band_percent']=float(np.mean(
        (sampled_hu[in_bounds]>=args.hu_low)&(sampled_hu[in_bounds]<=args.hu_high))*100)
    metrics['mesh_vertices_outside_ct']=int((~in_bounds).sum())
    metrics['preview_lighting']='Upper lights plus neutral lower fill; unchanged material across comparisons'
    heart_surface = make_surface(h,1.0)
    cor_surface = make_surface(cor,0.16)
    old_surface,_ = make_fat_surface(base_proxy)
    heart_actor = exterior.material_actor(heart_surface,(0.56,0.105,0.085),0.58,0.30)
    cor_actor = exterior.material_actor(cor_surface,(0.30,0.025,0.03),0.50,0.30)
    old_actor = exterior.material_actor(old_surface,(0.84,0.70,0.47),0.58,0.30)
    fat_actor = exterior.material_actor(fat_surface,(0.84,0.70,0.47),0.58,0.30)
    camera_config = json.loads((root/'data/ct/teachers/v4/manifest.json').read_text())['camera']
    center = np.asarray(camera_config['center'])
    radius = camera_config['radius']
    poses = legacy.make_poses(center,radius,120,42)
    rows=[]
    for view in (20,70,100):
        paths=[]
        for label, actors in [('old_proxy',[heart_actor,cor_actor,old_actor]),
                              ('corrected',[heart_actor,cor_actor,fat_actor]),
                              ('fat_only',[fat_actor])]:
            path=output/f'view_{view:03}_{label}.png'
            render_review(str(path),actors,poses[view],center,radius)
            paths.append(str(path))
        comparison=output/f'comparison_{view:03}.png'
        exterior.write_comparison(paths,['Old 4 mm rule (CT proxy)','Corrected CT-connected candidate','Corrected fat only'],str(comparison))
        rows.append(Image.open(comparison).convert('RGB'))
        print(f'Rendered review view {view}',flush=True)
    montage=Image.new('RGB',(rows[0].width,sum(r.height for r in rows)))
    y=0
    for row in rows:
        montage.paste(row,(0,y)); y+=row.height
    montage.save(output/'range_3d_comparison.png')

    counts=candidate.sum(axis=(1,2))
    nonempty=np.flatnonzero(counts)
    bins=np.array_split(nonempty,6)
    slices=[int(part[np.argmax(counts[part])]) for part in bins if len(part)]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    panels=[]
    for local_z in slices:
        gray=np.clip((hu[local_z].astype(float)+190)/490,0,1)
        rgb=np.repeat((gray*255).astype(np.uint8)[...,None],3,axis=2)
        # Green baseline, yellow added tissue, red removed baseline, cyan uncertain.
        for region,color in [(candidate[local_z]&base_proxy[local_z],(65,215,100)),
            ((candidate&~base_proxy)[local_z],(255,215,45)),
            ((base_proxy&~candidate)[local_z],(235,70,70)),
            (uncertain[local_z],(50,210,235))]:
            rgb[region]=(0.4*rgb[region]+0.6*np.asarray(color)).astype(np.uint8)
        boundary=h[local_z]&~erode2d(h[local_z])
        rgb[boundary]=(245,245,245)
        panel_img=Image.fromarray(rgb)
        width=440
        height=int(round(rgb.shape[0]*spacing[1]/(rgb.shape[1]*spacing[2])*width))
        panel_img=panel_img.resize((width,height),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(width,height+30),'#171717')
        panel.paste(panel_img,(0,30))
        ImageDraw.Draw(panel).text((8,5),f'Native axial z={local_z+lo[0]}',font=font,fill='white')
        panels.append(panel)
    width,height=panels[0].size
    axial=Image.new('RGB',(width*3,height*2+55),'#171717')
    text=ImageDraw.Draw(axial)
    text.text((8,6),'Green: retained  |  Yellow: added  |  Red: removed  |  Cyan: uncertain, excluded',font=font,fill='white')
    text.text((8,29),'Reference is reconstructed 4 mm HU proxy, not the unavailable old torso-fat mask.',font=font,fill='white')
    for n,panel in enumerate(panels):
        axial.paste(panel,((n%3)*width,(n//3)*height+55))
    axial.save(output/'ct_range_changes.png')
    metrics['review_slices_native_z']=[int(z+lo[0]) for z in slices]
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(f'Finished: {output}',flush=True)


if __name__=='__main__':
    main()
