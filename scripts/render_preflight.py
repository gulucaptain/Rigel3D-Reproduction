"""Run with Blender --background --disable-autoexec --python this-file.

One original TRAIN FBX multiview preflight; not a trained geometry decoder.
"""
import json
from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Vector
root=Path(__file__).resolve().parents[1];folder=root/'data/external/generation_preflight'
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(folder/'asset.fbx'),use_anim=False)
imported=list(bpy.context.scene.objects);meshes=[o for o in imported if o.type=='MESH']
if not meshes:raise ValueError('No original mesh')
bpy.context.view_layer.update()
corners=np.array([tuple(o.matrix_world@Vector(c)) for o in meshes for c in o.bound_box]);lo=corners.min(0);hi=corners.max(0);centre=(lo+hi)/2;extent=float((hi-lo).max())
holder=bpy.data.objects.new('NormalizedAsset',None);bpy.context.collection.objects.link(holder)
for obj in imported:
    if obj.parent is None:obj.parent=holder
holder.scale=(.9/extent,)*3;holder.location=Vector(-centre*.9/extent)
bpy.context.view_layer.update()
points=[]
deps=bpy.context.evaluated_depsgraph_get()
for obj in meshes:
    evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh()
    # The renderer evaluates the FBX armature/modifiers. Raw obj.data vertices
    # need not be in that pose; export the same evaluated geometry as rendered.
    points.extend(tuple(evaluated.matrix_world@v.co) for v in mesh.vertices)
    evaluated.to_mesh_clear()
np.savez_compressed(folder/'original_mesh_vertices.npz',points=np.asarray(points,dtype=np.float32))
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
scene.render.resolution_x=224;scene.render.resolution_y=224;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.8,.8,1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value=.8
light=bpy.data.lights.new('Area','AREA');light.energy=300;light.shape='DISK';light.size=3
obj=bpy.data.objects.new('Area',light);bpy.context.collection.objects.link(obj);obj.location=(2,-2,3)
camera=bpy.data.cameras.new('Camera');cam=bpy.data.objects.new('Camera',camera);bpy.context.collection.objects.link(cam);scene.camera=cam;camera.lens=35;camera.sensor_width=32
cameras=[];visibility=[];pixels=[]
for i,yaw in enumerate([0,90,180,270]):
    angle=math.radians(yaw);cam.location=(2*math.cos(angle),2*math.sin(angle),1.15)
    cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler();bpy.context.view_layer.update()
    scene.render.filepath=str(folder/f'view_{i}.png');bpy.ops.render.render(write_still=True)
    conversion=np.diag([1.,-1.,-1.,1.]);w2c=conversion@np.array(cam.matrix_world.inverted())
    cameras.append(dict(yaw=yaw,w2c=w2c.tolist(),K=[[245,0,112],[0,245,112],[0,0,1]],filename=f'view_{i}.png'))
    verts=np.asarray(points);campts=verts@w2c[:3,:3].T+w2c[:3,3]
    uv=campts[:,:2]/campts[:,2,None]*245+112;pixels.append(uv)
    vis=[];deps=bpy.context.evaluated_depsgraph_get()
    for vertex in verts:
        direction=Vector(vertex)-cam.location
        hit,location,*_=scene.ray_cast(deps,cam.location,direction.normalized())
        vis.append(bool(hit and (location-Vector(vertex)).length<2e-4))
    visibility.append(vis)
report=json.loads((root/'results/round1/generation_preflight.json').read_text())
report.update(blender=bpy.app.version_string,meshes=len(meshes),vertices=len(points),armatures=sum(o.type=='ARMATURE' for o in imported),views=4,
    render_scope='Original train asset, four 224x224 CPU Cycles views; explicit independent normalization; not author TRELLIS rendering or learned mesh reconstruction',
    normalization=dict(centre=centre.tolist(),longest_extent=extent,normalized_extent=.9))
(folder/'cameras.json').write_text(json.dumps(cameras,indent=2)+'\n')
np.savez_compressed(folder/'vertex_visibility.npz',pixels=np.asarray(pixels,dtype=np.float32),visible=np.asarray(visibility))
(root/'results/round1/generation_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('Original training mesh render preflight completed')
