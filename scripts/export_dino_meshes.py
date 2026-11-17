"""Blender-only: evaluate original asset poses and export data, never execute assets."""
import json
from pathlib import Path
import bpy
import numpy as np
root=Path(__file__).resolve().parents[1];manifest=json.loads((root/'results/round2/dino_cohort_sources.json').read_text());records=[]
axis=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
for rec in manifest['assets']:
    row={k:v for k,v in rec.items() if k not in ['storage_checks']}
    if rec['status']!='downloaded':records.append(row);continue
    dest=root/'data/external/dino_cohort'/Path(rec['filename']).stem/'raw_mesh.npz'
    if dest.exists():records.append(dict(**row,mesh_status='exported'));continue
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True);path=root/rec['raw_path'];suffix=path.suffix.lower()
        if suffix=='.fbx':bpy.ops.import_scene.fbx(filepath=str(path),use_anim=True)
        elif suffix in ['.glb','.gltf']:bpy.ops.import_scene.gltf(filepath=str(path))
        else:raise ValueError('Unsupported source format '+suffix)
        frame=rec['asset_name'].split('/')[-1];bpy.context.scene.frame_set(int(frame) if frame.isdigit() else 1)
        bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get();verts=[];faces=[];offset=0
        for obj in list(bpy.context.scene.objects):
            if obj.type!='MESH':continue
            ev=obj.evaluated_get(deps);mesh=ev.to_mesh();mesh.calc_loop_triangles()
            vv=np.array([tuple(ev.matrix_world@v.co) for v in mesh.vertices],dtype=float)@axis.T
            ff=np.array([list(t.vertices) for t in mesh.loop_triangles],dtype=np.int64).reshape(-1,3)
            verts.append(vv);faces.append(ff+offset);offset+=len(vv);ev.to_mesh_clear()
        if not verts:raise ValueError('No mesh')
        np.savez_compressed(dest,vertices=np.concatenate(verts).astype(np.float32),faces=np.concatenate(faces))
        records.append(dict(**row,mesh_status='exported',vertices=offset,frame=int(frame) if frame.isdigit() else 1))
    except Exception as e:records.append(dict(**row,mesh_status='failed',mesh_reason=type(e).__name__+': '+str(e)[:160]))
    print('Mesh export',rec['filename'],records[-1].get('mesh_status'),flush=True)
(root/'results/round2/mesh_export.json').write_text(json.dumps(dict(assets=records),indent=2)+'\n')
