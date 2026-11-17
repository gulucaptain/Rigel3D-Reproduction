"""Render only registration-gated meshes; export calibrated point visibility."""
import json,math
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
root=Path(__file__).resolve().parents[1];records=json.loads((root/'results/round2/mesh_alignment.json').read_text())['assets']
for row in records:
    if row.get('alignment_status')!='accepted':continue
    folder=root/'data/external/dino_cohort'/Path(row['filename']).stem
    if (folder/'projection.npz').exists():continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    d=np.load(folder/'aligned_mesh.npz');v=d['vertices'][:,[0,2,1]].copy();v[:,1]*=-1
    mesh=bpy.data.meshes.new('mesh');mesh.from_pydata(v.tolist(),[],d['faces'].tolist());mesh.update()
    obj=bpy.data.objects.new('asset',mesh);bpy.context.collection.objects.link(obj)
    material=bpy.data.materials.new('neutral');material.diffuse_color=(.58,.62,.68,1);obj.data.materials.append(material)
    scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True
    scene.display.shading.background_type='WORLD';scene.world=bpy.data.worlds.new('world');scene.world.color=(1,1,1)
    scene.render.resolution_x=224;scene.render.resolution_y=224;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
    camera=bpy.data.cameras.new('camera');cam=bpy.data.objects.new('camera',camera);bpy.context.collection.objects.link(cam);scene.camera=cam;camera.type='ORTHO';camera.ortho_scale=1.45
    points=np.load(root/'data/external/prepared'/row['filename'])['points'][:,[0,2,1]].copy();points[:,1]*=-1
    pixels=[];visible=[]
    for i,yaw in enumerate([0,90,180,270]):
        angle=math.radians(yaw);cam.location=(2*math.cos(angle),2*math.sin(angle),.75);cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()
        scene.render.filepath=str(folder/f'view_{i}.png');bpy.ops.render.render(write_still=True)
        deps=bpy.context.evaluated_depsgraph_get();uv=[];vis=[]
        # Orthographic rays share direction; start at the camera plane above each query.
        direction=cam.rotation_euler.to_matrix()@Vector((0,0,-1))
        for p in points:
            pp=Vector(p);ndc=world_to_camera_view(scene,cam,pp);uv.append((ndc.x*224,(1-ndc.y)*224))
            origin=pp-direction*3;hit,loc,*_=scene.ray_cast(deps,origin,direction)
            vis.append(bool(hit and (loc-pp).length<.02 and 0<=ndc.x<=1 and 0<=ndc.y<=1))
        pixels.append(uv);visible.append(vis)
    np.savez_compressed(folder/'projection.npz',pixels=np.asarray(pixels,np.float32),visible=np.asarray(visible))
    print('RENDERED',row['filename'],flush=True)
