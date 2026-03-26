from render import render_mesh

views = [
    ("front", 0, 0),
    ("left30", -30, 0),
    ("right30", 30, 0),
    ('left60', -60, 0),
    ("right60", 60, 0),
    ("top", 0, 30),
    ("down,", 0, -30)
]

for angle, azim, elev in views:
    render_mesh(
        obj_path="./assets/outputs/3d_reconstruct_obj/1602308_1_obj.obj", 
        out_path=f"./assets/outputs/3d_reconstruct_multi_view/{angle}.png", 
        azim=azim, 
        elev=elev, 
        dist=2.7
    )