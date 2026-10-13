"""Render saved metric reconstructions: three model rows, four scene columns.

No inference, GT fitting, or scale normalization. Missing predictions stay marked.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

SCENES = [
    ('v2x-radar-i','v2x-radar-i','2024-05-07-16-50-12','V2X-Radar-I',80.),
    ('waymo','waymo','validation_segment-7650923902987369309_2380_000_2400_000','Waymo',80.),
    ('JRDB','JRDB','memorial-court-2019-03-16_0','JRDB',40.),
    ('Humanoid_everyday','humanoid_everyday','Locomanip_throw_away_a_piece_of_paper_to_trashcan','Humanoid Everyday',5.),
]
MODELS=[('pi3','Pi3'),('vggt_omega','VGGT-Omega'),('4rc','4RC')]

def hom(a):
    a=np.asarray(a,float);t=np.eye(4);t[:a.shape[0],:a.shape[1]]=a;return t

def pose(path):
    return hom(json.loads(path.read_text())['extrinsics'])

def fuse(root, spec, model, frames, out):
    ds,dd,seq,_,max_depth=spec
    base=root/'outputs'/ds/seq/(model+'_outputs')/'optimized_scale_original'
    data=root/'data/official-data'/dd/seq
    cache=out/(ds+'_'+model+'_cloud.npz')
    if cache.exists():
        old_meta=json.loads((cache.with_suffix('.json')).read_text())
        if 'limited to top 40%' in old_meta.get('sky_filter',''):
            d=np.load(cache);return d['points'],d['colors'],old_meta
    e0=pose(base/'camera_params'/(frames[0]+'.json'))
    allp=[];allc=[];used=[];stats=[];rng=np.random.default_rng(20260919)
    for name in frames:
        path=base/'pointmap'/(name+'.npy');p=np.load(path,mmap_mode='r')
        h,w=p.shape[:2]
        rgb=np.asarray(Image.open(data/(name+'.jpg')).convert('RGB').resize((w,h),Image.Resampling.BILINEAR))
        # Remove discontinuity pixels (mixed foreground/background geometry).
        z=p[:,:,2];edge=np.zeros((h,w),bool)
        dx=np.abs(np.diff(z,axis=1))>np.maximum(.025,.06*np.minimum(z[:,:-1],z[:,1:]))
        dy=np.abs(np.diff(z,axis=0))>np.maximum(.025,.06*np.minimum(z[:-1],z[1:]))
        edge[:,:-1]|=dx;edge[:,1:]|=dx;edge[:-1]|=dy;edge[1:]|=dy
        valid=np.isfinite(p).all(2)&(z>.15)&(z<max_depth)&(~edge)
        # Conservative visualization-only sky heuristic: blue connected to top edge.
        col=rgb.astype(np.float32)
        blue=(col[:,:,2]>1.18*col[:,:,0])&(col[:,:,2]>1.025*col[:,:,1])&(col[:,:,2]>90)
        seed=np.zeros((h,w),bool);seed[:3]=blue[:3]
        sky=ndimage.binary_propagation(seed,mask=blue)
        sky[int(h*.40):]=False
        dynamic=np.zeros((h,w),bool)
        protected=np.zeros((h,w),bool)
        mask_dir=root/'outputs'/ds/seq/'sam_outputs'/name
        dynamic_classes={'pedestrian','person','car','van','truck','bus','bicycle','cyclist','motorcycle','motorcyclist'}
        for mf in mask_dir.glob('*_mask_*.png'):
            if mf.name.split('_mask_')[0].lower() in {'road','ground','floor','curbstone','sidewalk'}:
                protected|=np.asarray(Image.open(mf).convert('L').resize((w,h),Image.Resampling.NEAREST))>127
            if mf.name.split('_mask_')[0].lower() in dynamic_classes:
                dynamic|=np.asarray(Image.open(mf).convert('L').resize((w,h),Image.Resampling.NEAREST))>127
        sky &= ~protected
        valid &= ~(sky|dynamic)
        # Uniform image sampling; do not use possibly stale relative confidence maps.
        sample=np.zeros((h,w),bool);sample[::3,::3]=True;valid &= sample
        ids=np.flatnonzero(valid)
        if len(ids)>14000:ids=rng.choice(ids,14000,replace=False)
        pts=np.asarray(p).reshape(-1,3)[ids];colors=rgb.reshape(-1,3)[ids]
        t=e0@np.linalg.inv(pose(base/'camera_params'/(name+'.json')))
        pts=pts@t[:3,:3].T+t[:3,3]
        allp.append(pts.astype(np.float32));allc.append(colors);used.append(name)
        stats.append({'frame':name,'points':len(pts),'sky_pixels':int(sky.sum()),'dynamic_mask_pixels':int(dynamic.sum()),'pointmap_mtime_ns':path.stat().st_mtime_ns})
    points=np.concatenate(allp);colors=np.concatenate(allc)
    # One fixed voxel size per dataset for all models; units remain meters.
    voxel=.06 if max_depth>=40 else .006
    _,idx=np.unique(np.floor(points/voxel).astype(np.int64),axis=0,return_index=True)
    if len(idx)>500000:idx=rng.choice(idx,500000,replace=False)
    points,colors=points[idx],colors[idx]
    meta={'dataset':ds,'sequence':seq,'model':model,'variant':'optimized_scale_original',
          'source':str(base),'frame_names':used,'frame_count':len(used),'points':len(points),
          'coordinate_frame':'first predicted camera; x right, y down, z forward',
          'fusion':'E_first @ inverse(E_frame); saved world-to-camera extrinsics',
          'depth_range_m':[.15,max_depth],'pixel_stride':3,'voxel_size_m':voxel,
          'confidence_filter':False,'discontinuity_relative_threshold':.06,
          'dynamic_motion_compensation':False,'dynamic_sam_mask_exclusion':sorted(dynamic_classes),
          'sky_filter':'top-edge-connected B>1.18R, B>1.025G, B>90; limited to top 40%; road/ground/floor/curbstone/sidewalk SAM regions protected; visualization heuristic',
          'scale_fitting':False,'files':stats}
    np.savez_compressed(cache,points=points,colors=colors)
    cache.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    return points,colors,meta

def project(p,elevation):
    # Horizontal u is camera-forward z, v mixes camera-right x and camera-up -y.
    el=np.deg2rad(elevation)
    xy=np.c_[p[:,2],np.sin(el)*p[:,0]-np.cos(el)*p[:,1]]
    toward=np.cos(el)*p[:,0]+np.sin(el)*p[:,1]
    return xy,toward

def render(cases,out,elevation,compact=False):
    cols=[i for i in range(4) if any((j,i) in cases for j in range(3))] if compact else list(range(4))
    fig,axes=plt.subplots(3,len(cols),figsize=(5.0*len(cols)+1,13.6),squeeze=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.11,top=.84,wspace=.22,hspace=.24)
    limits={}
    for i in cols:
        clouds=[project(cases[j,i][0],elevation)[0] for j in range(3) if (j,i) in cases]
        if not clouds:continue
        lo=np.min([np.quantile(p,.008,axis=0) for p in clouds],axis=0)
        hi=np.max([np.quantile(p,.992,axis=0) for p in clouds],axis=0)
        center=(lo+hi)/2;span=max(hi-lo)*1.08;limits[i]=(center-span/2,center+span/2)
    for c,i in enumerate(cols):
        ds,dd,seq,label,_=SCENES[i]
        short={'waymo':'7650923902987369309\n2380_000–2400_000','JRDB':'memorial-court\n2019-03-16_0',
               'Humanoid_everyday':'Throw a piece of paper\ninto the trashcan'}.get(ds,seq)
        axes[0,c].set_title(label+'\n'+short,fontsize=13,weight='bold',pad=15)
        for j,(_,model_label) in enumerate(MODELS):
            ax=axes[j,c];ax.set_facecolor('white');ax.grid(color='#dce1e6',lw=.55);ax.set_axisbelow(True)
            ax.tick_params(labelsize=9);ax.set_box_aspect(1)
            for spine in ax.spines.values():spine.set_color('#c1c9d1')
            if c==0:
                box=ax.get_position();fig.text(.018,(box.y0+box.y1)/2,model_label,rotation=90,va='center',ha='center',fontsize=18,weight='bold')
            if (j,i) not in cases:
                ax.text(.5,.53,'Prediction files unavailable',ha='center',va='center',transform=ax.transAxes,color='#7a818b',fontsize=11)
                ax.text(.5,.43,'Specified sequence retained',ha='center',transform=ax.transAxes,color='#8b9199',fontsize=9)
                ax.set_xticks([]);ax.set_yticks([]);continue
            p,rgb,meta=cases[j,i];xy,order_z=project(p,elevation);low,high=limits[i]
            valid=(xy>=low).all(1)&(xy<=high).all(1);idx=np.flatnonzero(valid);idx=idx[np.argsort(order_z[idx])[::-1]]
            ax.scatter(xy[idx,0],xy[idx,1],c=rgb[idx]/255.,s=.32,linewidths=0,rasterized=True)
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal')
            ax.set_xlabel('z (m)' if elevation==90 else 'u (m)',fontsize=10)
            ax.set_ylabel('x (m)' if elevation==90 else 'v (m)',fontsize=10)
            ax.text(.025,.975,f"{meta['frame_count']} frames",transform=ax.transAxes,va='top',fontsize=9,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    tag='bev' if elevation==90 else 'oblique'
    title='Metric 3D reconstruction | optimized_scale_original'
    fig.suptitle(title,fontsize=21,weight='bold',y=.97)
    fig.text(.5,.935,'RGB point clouds fused using each model\'s predicted poses',ha='center',fontsize=12,color='#555b65')
    foot='Same metric limits and view within each column; no scale normalization. '
    foot+=('First-camera x–z projection.' if elevation==90 else f'Orthographic projection, elevation {elevation:g}°; u = z, v = sin(elevation) x − cos(elevation) y.')
    fig.text(.075,.051,foot,fontsize=10,color='#58606a')
    fig.text(.075,.029,'Visualization filters: depth edges, top-connected blue sky, available dynamic-object SAM masks, voxel subsampling.',fontsize=10,color='#58606a')
    name='reconstruction_'+tag+('_available' if compact else '_four_sequences')
    fig.savefig(out/(name+'.png'),dpi=220,facecolor='white')
    fig.savefig(out/(name+'.pdf'),dpi=180,facecolor='white')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);cases={};manifest=[]
    for i,spec in enumerate(SCENES):
        ds,dd,seq,_,_=spec;sets=[]
        for m,_ in MODELS:
            b=args.project/'outputs'/ds/seq/(m+'_outputs')/'optimized_scale_original'
            names={f.stem for f in (b/'pointmap').glob('*.npy') if (b/'camera_params'/(f.stem+'.json')).exists() and (args.project/'data/official-data'/dd/seq/(f.stem+'.jpg')).exists()}
            sets.append(names)
        common=sorted(set.intersection(*sets))
        if not common:
            manifest.append({'dataset':ds,'sequence':seq,'status':'missing predictions','available_frames':dict(zip([m for m,_ in MODELS],map(len,sets)))});continue
        for j,(m,_) in enumerate(MODELS):
            print('FUSING',ds,m,len(common),flush=True)
            cases[j,i]=fuse(args.project,spec,m,common,out);manifest.append(cases[j,i][2])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for el in [65,90]:
        for compact in [False,True]:render(cases,out,el,compact)
    print('DONE',out,flush=True)

if __name__=='__main__':main()
