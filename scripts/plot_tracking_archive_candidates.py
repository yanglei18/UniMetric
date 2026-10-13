import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
root=Path("analysis_outputs/kitti-tracking_three_model_selection")
ds=json.loads((root/"archive_gt.json").read_text(encoding="utf-8-sig"))
order=["testing_0012","testing_0016","training_0001"]
fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
for ax,name in zip(axes,order):
 d=next((r for r in ds if r["sequence"]==name),None)
 if not d:
  ax.text(.5,.5,"GT unavailable",transform=ax.transAxes,ha="center");continue
 p=np.array(d["xy"]);ax.plot(p[:,0],p[:,1],color="#078477",lw=2);ax.scatter(p[0,0],p[0,1],c="#d94a35",s=30,zorder=3)
 ax.set(title=f"{name} | {d['frames']} frames | {d['path_m']:.0f} m",xlabel="x (m)",ylabel="y (m)",aspect="equal");ax.grid(alpha=.25)
fig.suptitle("KITTI-tracking candidates | GT trajectories; red dot = start")
fig.savefig(root/"new_tracking_gt_candidates.png",dpi=180)
