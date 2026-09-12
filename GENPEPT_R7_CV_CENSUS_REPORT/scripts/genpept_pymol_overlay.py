from __future__ import annotations

import json

from pymol import cmd

WORK = "/tmp/opencode/overlay_seeds"
panels = json.load(open(f"{WORK}/panels.json"))

cmd.set("ray_opaque_background", 0)
cmd.bg_color("white")
cmd.set("antialias", 2)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_highlight_color", "grey50")
cmd.set("orthoscopic", "on")
cmd.set("ray_trace_mode", 0)

for p in panels:
    cmd.delete("all")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("orthoscopic", "on")
    cmd.load(p["native_pdb"], "nat")
    cmd.load(p["seed_pdb"], "seed")
    cmd.hide("everything", "all")
    cmd.show("sticks", "(nat or seed) and not elem H")
    cmd.color("firebrick", "nat")
    cmd.color("marine", "seed and not elem H")
    cmd.set("stick_transparency", 0.45, "seed")
    cmd.set("stick_radius", 0.12)
    cmd.set("sphere_scale", 0.18)
    cmd.show("spheres", "nat and elem C")
    cmd.color("firebrick", "nat")
    cmd.orient("nat")
    cmd.zoom("nat", buffer=4)
    cmd.ray(640, 480)
    cmd.png(p["out_png"], dpi=150)

cmd.quit()
