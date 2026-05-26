from pathlib import Path

_IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}
_MESH_FORMATS = {".stl", ".vtp", ".vtk", ".ply", ".obj"}
_ANIMATION_FORMATS = {".gif", ".mp4"}


def export_frames(output, time_indices, update_frame, mesh, plotter):
    path = Path(output)
    ext = path.suffix.lower()
    multi = len(time_indices) > 1

    if ext in _ANIMATION_FORMATS:
        if ext == ".gif":
            plotter.open_gif(str(path))
        else:
            plotter.open_movie(str(path))
        for ti in time_indices:
            update_frame(ti)
            plotter.write_frame()
        plotter.close()
    elif multi:
        for i, ti in enumerate(time_indices):
            update_frame(ti)
            dest = path.parent / f"{path.stem}_{i:04d}{path.suffix}"
            if ext in _IMAGE_FORMATS:
                plotter.screenshot(str(dest))
            elif ext in _MESH_FORMATS:
                mesh.save(str(dest))
    else:
        update_frame(time_indices[0])
        if ext in _IMAGE_FORMATS:
            plotter.screenshot(str(path))
        elif ext in _MESH_FORMATS:
            mesh.save(str(path))
