"""Actual depth-grid triangles, CPU z-buffer, shared views and local detail insets."""
from pathlib import Path
import argparse,json
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,ConnectionPatch

SEQ='validation_segment-7650923902987369309_2380_000_2400_000'
MODELS=[('pi3','Pi3'),('vggt_omega','VGGT-Omega'),('4rc','4RC')]

def E(p):
    a=np.asarray(json.loads(p.read_text())['extrinsics'],float);t=np.eye(4);t[:a.shape[0]]=a;return t

def norm(p):return p/np.linalg.norm(p)

@njit(cache=True)
def raster(vertices,colors,faces,bounds,width,height):
    image=np.full((height,width,3),255,np.uint8);depth=np.full((height,width),np.inf,np.float32)
    sx=(width-1)/(bounds[1]-bounds[0]);sy=(height-1)/(bounds[3]-bounds[2])
    for f in faces:
        a,b,c=f
        x0=(vertices[a,0]-bounds[0])*sx;y0=(bounds[3]-vertices[a,1])*sy
        x1=(vertices[b,0]-bounds[0])*sx;y1=(bounds[3]-vertices[b,1])*sy
        x2=(vertices[c,0]-bounds[0])*sx;y2=(bounds[3]-vertices[c,1])*sy
        xmin=max(0,int(np.floor(min(x0,x1,x2))));xmax=min(width-1,int(np.ceil(max(x0,x1,x2))))
        ymin=max(0,int(np.floor(min(y0,y1,y2))));ymax=min(height-1,int(np.ceil(max(y0,y1,y2))))
        if xmin>xmax or ymin>ymax:continue
        det=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(det)<1e-8:continue
        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                w0=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/det
                w1=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/det;w2=1-w0-w1
                if w0<0 or w1<0 or w2<0:continue
                z=w0*vertices[a,2]+w1*vertices[b,2]+w2*vertices[c,2]
                if z<depth[y,x]:
                    depth[y,x]=z
                    for k in range(3):image[y,x,k]=min(255,max(0,int(w0*colors[a,k]+w1*colors[b,k]+w2*colors[c,k])))
    return image

def load_mesh(root,model,out):
    base=root/'outputs/waymo'/SEQ/(model+'_outputs/optimized_scale_original');data=root/'data/official-data/waymo'/SEQ
    files=sorted((base/'pointmap').glob('*.npy'));ref=E(base/'camera_params'/(files[0].stem+'.json'));pts=[];cols=[];faces=[];meta=[];offset=0;trajectory=[]
    for f in files:
        p=np.load(f);h,w=p.shape[:2];rgb=np.asarray(Image.open(data/(f.stem+'.jpg')).convert('RGB').resize((w,h),Image.Resampling.BILINEAR));col=rgb.astype(float)
        blue=(col[:,:,2]>col[:,:,0]*1.18)&(col[:,:,2]>col[:,:,1]*1.025)&(col[:,:,2]>90);seed=np.zeros((h,w),bool);seed[:3]=blue[:3];sky=ndimage.binary_propagation(seed,mask=blue)
        # A blue cast on asphalt can connect to blue sky through the image border.
        # Never propagate the heuristic into the lower image or known road masks.
        sky[int(h*.40):]=False
        dynamic=np.zeros((h,w),bool);road=np.zeros((h,w),bool)
        for mask in (root/'outputs/waymo'/SEQ/'sam_outputs'/f.stem).glob('*_mask_*.png'):
            label=mask.name.split('_mask_')[0].lower()
            if label in {'road','ground','floor','curbstone','sidewalk'}:
                road|=np.asarray(Image.open(mask).convert('L').resize((w,h),Image.Resampling.NEAREST))>127
            if label in {'car','suv','pickup truck','van','bus','truck','person','pedestrian','cyclist','bicycle','motorcycle'}:
                dynamic|=np.asarray(Image.open(mask).convert('L').resize((w,h),Image.Resampling.NEAREST))>127
        sky &= ~road
        # Retain a dense grid, but never join triangles across invalid pixels or large depth jumps.
        stride=3;valid=np.isfinite(p).all(2)&(p[:,:,2]>.3)&(p[:,:,2]<45)&~(sky|dynamic)
        # Explicit road-focused display ROI, shared across frames/models. This
        # excludes uncertain sky/upper facades rather than presenting them as a map.
        valid[:int(h*.38)]=False
        safe=ndimage.minimum_filter(valid.astype(np.uint8),size=stride+1)>0
        q=p[::stride,::stride];color=rgb[::stride,::stride];good=safe[::stride,::stride];gh,gw=good.shape;ids=np.arange(gh*gw).reshape(gh,gw)
        tri=np.concatenate([np.stack([ids[:-1,:-1].ravel(),ids[1:,:-1].ravel(),ids[:-1,1:].ravel()],axis=1),np.stack([ids[1:,1:].ravel(),ids[:-1,1:].ravel(),ids[1:,:-1].ravel()],axis=1)])
        flat=q.reshape(-1,3);z=flat[:,2];keep=good.ravel()[tri].all(1)
        zrange=np.ptp(z[tri],axis=1);keep&=zrange<np.maximum(.12,.07*np.min(z[tri],axis=1))
        edge=np.maximum.reduce([np.linalg.norm(flat[tri[:,0]]-flat[tri[:,1]],axis=1),np.linalg.norm(flat[tri[:,0]]-flat[tri[:,2]],axis=1),np.linalg.norm(flat[tri[:,1]]-flat[tri[:,2]],axis=1)])
        keep&=edge<1.2;tri=tri[keep]
        t=ref@np.linalg.inv(E(base/'camera_params'/(f.stem+'.json')));world=flat@t[:3,:3].T+t[:3,3];trajectory.append(t[:3,3])
        pts.append(np.nan_to_num(world).astype(np.float32));cols.append(color.reshape(-1,3));faces.append((tri+offset).astype(np.int32));offset+=len(flat)
        meta.append({'frame':f.stem,'faces':len(tri),'pointmap':str(f),'mtime_ns':f.stat().st_mtime_ns,'sky_fraction':float(sky.mean()),'lower_half_sky_pixels':int(sky[h//2:].sum()),'road_protected_pixels':int(road.sum()),'display_image_roi_y_fraction':[.38,1.]})
    mesh=(np.concatenate(pts),np.concatenate(cols),np.concatenate(faces),np.asarray(trajectory),len(pts[0]),len(faces[0]))
    (out/(model+'_source.json')).write_text(json.dumps(meta,indent=2));return mesh

def project(p,basis,el=48):
    q=p@basis.T;e=np.deg2rad(el)
    return np.c_[q[:,0],np.sin(e)*q[:,1]+np.cos(e)*q[:,2],np.cos(e)*q[:,1]-np.sin(e)*q[:,2]].astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.project;out=args.output;out.mkdir(parents=True,exist_ok=True)
    meshes={}
    for m,_ in MODELS:print('LOAD',m,flush=True);meshes[m]=load_mesh(root,m,out)
    first=sorted((root/'data/official-data/waymo'/SEQ).glob('*.yaml'))[0];cal=yaml.safe_load(first.read_text());up=norm(np.array(cal['camera1']['extrinsic'])[:3,2]);t=meshes['pi3'][3];along=t[-1]-t[0];along=norm(along-up*(along@up));side=norm(np.cross(up,along));basis=np.stack([along,side,up]);assert np.allclose(basis@basis.T,np.eye(3))
    projected={m:project(meshes[m][0],basis) for m,_ in MODELS};limits=[]
    for m,_ in MODELS:
        used=np.unique(meshes[m][2]);limits.append(np.quantile(projected[m][used,:2],[.005,.995],axis=0))
    lo=np.min([x[0] for x in limits],axis=0);hi=np.max([x[1] for x in limits],axis=0);pad=(hi-lo)*.025;lo-=pad;hi+=pad;bounds=np.array([lo[0],hi[0],lo[1],hi[1]])
    # Shared metric image aspect and two scene regions selected once for all models.
    span=hi-lo;center=(lo+hi)/2
    if span[0]/span[1]<2.55:
        half=span[1]*2.55/2;bounds[:2]=[center[0]-half,center[0]+half]
    else:
        half=span[0]/2.55/2;bounds[2:]=[center[1]-half,center[1]+half]
    route=project(t,basis);centers=[route[int(len(route)*v),:2]+np.array([0,-2.5]) for v in [.25,.65]]
    boxes=[np.array([c[0]-9,c[0]+9,c[1]-9,c[1]+9]) for c in centers]
    fig=plt.figure(figsize=(22,11.8));grid=fig.add_gridspec(3,3,width_ratios=[2.55,1,1],left=.06,right=.99,top=.88,bottom=.1,hspace=.25,wspace=.18)
    for row,(m,label) in enumerate(MODELS):
        vertices=projected[m];colors=meshes[m][1];faces=meshes[m][2];print('RENDER',m,'faces',len(faces),flush=True)
        ax=fig.add_subplot(grid[row,0]);im=raster(vertices,colors,faces,bounds,1600,628);Image.fromarray(im).save(out/(m+'_overview.png'));ax.imshow(im,extent=bounds,origin='upper');ax.set_aspect('equal');ax.set(xlabel='u (m)',ylabel='v (m)');pos=ax.get_position();fig.text(.015,(pos.y0+pos.y1)/2,label,rotation=90,ha='center',va='center',fontsize=16,weight='bold')
        if row==0:ax.set_title('Waymo | 40-frame metric reconstruction',weight='bold',fontsize=15)
        for k,bb in enumerate(boxes):
            detail=fig.add_subplot(grid[row,k+1]);img=raster(vertices,colors,faces,bb,650,650);detail.imshow(img,extent=bb,origin='upper');detail.set_aspect('equal');detail.set(xlabel='u (m)',ylabel='v (m)');detail.set_title('Detail '+str(k+1),fontsize=12)
            ax.add_patch(Rectangle((bb[0],bb[2]),bb[1]-bb[0],bb[3]-bb[2],fill=False,ec='#35679d',lw=1.2,ls='--'))
            ax.text((bb[0]+bb[1])/2,bb[3]+.7,str(k+1),ha='center',color='#35679d',fontsize=10)
        # Independent check: first-frame geometry versus full-fusion geometry.
        diag_bounds=boxes[0];single_faces=faces[:meshes[m][5]]
        one=raster(vertices,colors,single_faces,bounds,1600,628);Image.fromarray(one).save(out/(m+'_first_frame_same_view.png'))
    fig.suptitle('Waymo | '+SEQ.replace('validation_segment-',''),fontsize=20,weight='bold',y=.965)
    fig.text(.5,.922,'Road-focused view · saved optimized_scale_original + predicted poses · depth-grid surface rendering',ha='center',fontsize=12,color='#555')
    fig.text(.06,.046,'Same road-aligned view and meter scale across models. Triangles interpolate neighboring valid depth samples; no learned completion or pose refinement.',fontsize=10,color='#555')
    fig.text(.06,.023,'Display ROI: lower 62% of every image; depth 0.3–45 m. Road masks protected; sky/dynamic and discontinuity filters applied. Upper scene omitted.',fontsize=10,color='#555')
    for ext in ['png','pdf']:fig.savefig(out/('waymo_surface_details.'+ext),dpi=200,facecolor='white')
    im=Image.open(out/'waymo_surface_details.png');im.thumbnail((2400,1500));im.convert('RGB').save(out/'waymo_surface_details_preview.jpg',quality=90)
    (out/'render_settings.json').write_text(json.dumps({'sequence':SEQ,'variant':'optimized_scale_original','basis':basis.tolist(),'elevation_degrees':48,'bounds':bounds.tolist(),'detail_bounds':[b.tolist() for b in boxes],'depth_max_m':45,'pixel_stride':3,'max_triangle_edge_m':1.2,'no_scale_fitting':True,'no_pose_refinement':True},indent=2))
    print('DONE',out,flush=True)

if __name__=='__main__':main()
