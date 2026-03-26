import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)


device = torch.device("cpu")

DATA_DIR = './assets/outputs/3d_reconstruct_obj'
obj_path = os.path.join(DATA_DIR, "24795717_1_obj_1.obj")

OUT_DIR = "./assets/outputs/3d_reconstruct_render"
out_path = os.path.join(OUT_DIR, "24795717_1_render_1.png")

#create a render
def create_render(dist=5.0, elev=0, azim=0):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer

def render_mesh(obj_path=obj_path, out_path=out_path, azim=0, elev=0, dist=5.0):
    #Load obj file
    verts, faces_idx, _ = load_obj(obj_path)
    faces = faces_idx.verts_idx

    center = verts.mean(0)
    verts = verts - center
    scale = verts.abs().max()
    verts = verts / scale

    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    renderer = create_render(dist=dist, elev=elev, azim=azim)
    images = renderer(mesh)
    image = images[0, ..., :3].cpu().numpy()
    
    plt.imsave(out_path, image)

if __name__ == "__main__":
    render_mesh(dist=2.7)