"""Gravity-oriented 3x4 reconstruction figure, reusing verified fused caches."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_requested_reconstruction_grid import SCENES, MODELS, fuse, pose


def normalize(v):
    return v/np.linalg.norm(v)


def gravity(root,spec,frames):
    ds,dd,seq,_,_=spec
    if ds!='Humanoid_everyday':
        data=yaml.safe_load((root/'data/official-data'/dd/seq/(frames[0].split('_camera')[0]+'.yaml')).read_text())
        n=normalize(np.asarray(data['camera1']['extrinsic'])[:3,2])
        return n,{'source':'first-frame camera_from_lidar rotation applied to lidar +Z','normal_camera0':n.tolist()}
    # Ground masks, not arbitrary camera up: this camera starts pointed down.
    base=root/'outputs'/ds/seq/'pi3_outputs/optimized_scale_original'
    ref=pose(base/'camera_params'/(frames[0]+'.json'))
    normals=[];info=[];o3d.utility.random.seed(19)
    for mf in sorted((root/'outputs'/ds/seq/'sam_outputs').glob('*/floor_mask*.png')):
        name=mf.parent.name;pf=base/'pointmap'/(name+'.npy')
        if not pf.exists():continue
        p=np.load(pf);h,w=p.shape[:2]
        mask=np.asarray(Image.open(mf).convert('L').resize((w,h),Image.Resampling.NEAREST))>127
        valid=mask&np.isfinite(p).all(2)&(p[:,:,2]>.1)&(p[:,:,2]<5)
        pts=p[valid][::3]
        if len(pts)<100:continue
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        plane,ids=cloud.segment_plane(.01,3,500)
        if len(ids)/len(pts)<.7:continue
        n=normalize(np.array(plane[:3]))
        if n@pts.mean(0)>0:n=-n
        n=(ref@np.linalg.inv(pose(base/'camera_params'/(name+'.json'))))[:3,:3]@n
        normals.append(n);info.append({'frame':name,'normal_camera0':n.tolist(),'inlier_fraction':len(ids)/len(pts)})
    if not normals:raise RuntimeError('No reliable floor normal for Humanoid')
    n=normalize(np.median(normals,axis=0))
    return n,{'source':'median floor normals from Pi3 saved geometry; ONE common orientation for all models',
              'normal_camera0':n.tolist(),'samples':info}


def basis(up):
    right=np.array([1.,0,0]);right=normalize(right-up*(right@up))
    forward=normalize(np.cross(up,right))
    b=np.stack([right,forward,up]);assert np.allclose(b@b.T,np.eye(3),atol=1e-5)
    assert np.linalg.det(b)>.999
    return b


def projection(points, az, el):
    az,el=np.deg2rad([az,el]);x,y,z=points.T
    u=np.cos(az)*x+np.sin(az)*y
    horizontal_depth=-np.sin(az)*x+np.cos(az)*y
    v=np.sin(el)*horizontal_depth+np.cos(el)*z
    view_depth=np.cos(el)*horizontal_depth-np.sin(el)*z
    return np.c_[u,v],view_depth


def render(cases,bases,views,out,bev=False):
    fig,axes=plt.subplots(3,4,figsize=(21,13.6))
    fig.subplots_adjust(left=.075,right=.985,bottom=.10,top=.84,wspace=.20,hspace=.22)
    short=['2024-05-07-16-50-12','7650923902987369309\n2380_000–2400_000','memorial-court\n2019-03-16_0','Throw a piece of paper\ninto the trashcan']
    limits={};transformed={}
    for i in range(4):
        az,el=(0,90) if bev else views[i]
        allxy=[]
        for j in range(3):
            p,c,m=cases[j,i];q=p@bases[i].T
            xy,depth=projection(q,az,el);transformed[j,i]=(xy,depth,c,m)
            allxy.append(xy)
        lo=np.min([np.quantile(p,.005,axis=0) for p in allxy],axis=0)
        hi=np.max([np.quantile(p,.995,axis=0) for p in allxy],axis=0)
        center=(lo+hi)/2;half=max(hi-lo)*.535;limits[i]=(center-half,center+half)
    for i,spec in enumerate(SCENES):
        axes[0,i].set_title(spec[3]+'\n'+short[i],fontsize=13,weight='bold',pad=13)
        for j,(_,label) in enumerate(MODELS):
            ax=axes[j,i];xy,depth,c,meta=transformed[j,i];lo,hi=limits[i]
            idx=np.flatnonzero((xy>=lo).all(1)&(xy<=hi).all(1));idx=idx[np.argsort(depth[idx])[::-1]]
            ax.set_axisbelow(True);ax.grid(color='#e2e6eb',lw=.6)
            ax.scatter(xy[idx,0],xy[idx,1],c=c[idx]/255.,s=.43,linewidths=0,rasterized=True)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='X (m)' if bev else 'u (m)',ylabel='Y (m)' if bev else 'v (m)')
            ax.set_box_aspect(1);ax.tick_params(labelsize=9)
            for spine in ax.spines.values():spine.set_color('#bfc7ce')
            ax.text(.025,.975,f"{meta['frame_count']} frames",transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
            if i==0:
                pos=ax.get_position();fig.text(.022,(pos.y0+pos.y1)/2,label,rotation=90,va='center',ha='center',fontsize=18,weight='bold')
    fig.suptitle('Metric 3D reconstruction | '+('Ground-oriented top view' if bev else 'Upright oblique view'),fontsize=22,weight='bold',y=.975)
    fig.text(.5,.939,'Saved optimized_scale_original · RGB point clouds fused with predicted poses',ha='center',fontsize=12,color='#525b66')
    fig.text(.075,.050,'Same scene orientation, viewpoint and metric limits across models. Rigid rotation only; no scale fitting.',fontsize=11,color='#525b66')
    fig.text(.075,.027,'Up: sensor calibration (outdoor/JRDB), fitted floor normal (Humanoid). Sky, depth edges and available dynamic masks filtered.',fontsize=10,color='#525b66')
    stem='reconstruction_upright_bev' if bev else 'reconstruction_upright_oblique'
    for ext in ['png','pdf']:fig.savefig(out/(stem+'.'+ext),dpi=220 if ext=='png' else 180,facecolor='white')
    im=Image.open(out/(stem+'.png'));im.thumbnail((2200,1500));im.convert('RGB').save(out/(stem+'_preview.jpg'),quality=92)
    plt.close(fig)
    return {SCENES[i][0]:{'low':limits[i][0].tolist(),'high':limits[i][1].tolist()} for i in limits}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--previous-cache',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);cases={};bases={};manifest=[];orientation={}
    # Each scene's fixed virtual camera is shared by its three models.
    views=[(-25,55),(-30,55),(-25,48),(0,22)]
    for i,spec in enumerate(SCENES):
        ds,dd,seq,_,_=spec;names=[]
        for m,_ in MODELS:
            b=args.project/'outputs'/ds/seq/(m+'_outputs')/'optimized_scale_original'
            names.append({f.stem for f in (b/'pointmap').glob('*.npy') if (b/'camera_params'/(f.stem+'.json')).exists() and (args.project/'data/official-data'/dd/seq/(f.stem+'.jpg')).exists()})
        frames=sorted(set.intersection(*names))
        if not frames:raise RuntimeError('Missing complete predictions: '+ds)
        up,orient=gravity(args.project,spec,frames);bases[i]=basis(up)
        orientation[ds]=dict(orient,basis_rows_right_forward_up=bases[i].tolist(),azimuth_elevation_deg=views[i])
        for j,(m,_) in enumerate(MODELS):
            cache=args.previous_cache/(ds+'_'+m+'_cloud.npz');meta_path=cache.with_suffix('.json');reuse=False
            if cache.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text());base=args.project/'outputs'/ds/seq/(m+'_outputs/optimized_scale_original/pointmap')
                reuse='limited to top 40%' in meta.get('sky_filter','') and meta['frame_names']==frames and all((base/(x['frame']+'.npy')).stat().st_mtime_ns==x['pointmap_mtime_ns'] for x in meta['files'])
            if reuse:
                d=np.load(cache);cases[j,i]=(d['points'],d['colors'],meta)
            else:
                print('FUSING',ds,m,len(frames),flush=True);cases[j,i]=fuse(args.project,spec,m,frames,out)
            manifest.append(dict(cases[j,i][2],reused_cache=str(cache) if reuse else None))
            print('READY',ds,m,'cache',reuse,flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    orientation['display_limits_oblique']=render(cases,bases,views,out)
    orientation['display_limits_bev']=render(cases,bases,views,out,bev=True)
    (out/'orientation.json').write_text(json.dumps(orientation,indent=2))
    print('DONE',out,flush=True)

if __name__=='__main__':main()
