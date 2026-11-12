#!/usr/bin/env python3
"""FaceASI Measure v62 — OAK-D Pro / active IR stereo compatibility, preserving v58 metrics.
Run: python Python_FaceAssymmetry.py [--camera-check] [--ir-dot 0.35] [--ir-flood 0.10]
Dependencies on device: existing Kivy 2.3.1, NumPy, OpenCV, MediaPipe 0.10.21,
and existing DepthAI v3 (v2 adapter included). No MATLAB/Open3D/Matplotlib.
v59: OAK-D Pro IR dot/flood controls, camera diagnostics, camera/IR metadata.
v58: Direct Export CSV folder picker, remembered selection, existing sync-folder discovery,
and explicit local results fallback.
v57: Small current-frame expression estimate from 2D eye/brow/lip ratios.
Use Capture > Face points > Calibrate relaxed face (15 frames) for each person.
Eye-opening/brow-motion checks are separate from the unchanged 3D ASI.
Heuristic labels are not a trained classifier or validation of depth accuracy.
Scan result shows one whole-recording ASI plus upper/lower results.
Live and frame playback show one smoothed value; the averaging-window popup
has large buttons including 60 frames. Distances groups nose/mouth and brow.
CSV exports include a scan_summary row in addition to per-frame results.
Dense landmarks are optional in Capture settings and Review; valid stereo XYZ
and 2D FaceMesh indices are exported separately from the 21 anatomical IDs.
Photos are optional (--save-photos), and the default review opens in 3D.
The MATLAB CSV contains raw/filtered regional ASIs and dense_landmark records;
MATLAB viewers must select record_type when reading this long-format CSV.
Research software; 30–40 cm is a configurable experimental target, not a validated optimum.
"""
import argparse, csv, math, os, sys, threading, time, traceback
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np

VERSION='62.0'
AUTO_CSV_DIR=Path('/home/face/FaceASIScope/results')
DEFAULT_SMOOTHING_N=5
SMOOTHING_MAX_N=90
TARGET_MM=350.0
TOLERANCE_MM=50.0
FRAME_SIZE=(640,480)
MAX_RENDER_POINTS=32000
CLOUD_STRIDE=1
STALE_SECONDS=0.8
SCRIPT_DIR=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get('FACEASI_DATA_DIR',str(SCRIPT_DIR/'results'/'single_captures')))
CAPTURE_MODES=('Neutral','Smiling','Blinking')
# Exactly the 21 originally exported automatic IDs; do not claim the other
# requested anatomy (true trichion, eyelid margins, Cupid's bow) is captured.
LM={
 'GLABELLA':9,'SELLION_NASION':168,'PRONASALE':4,'SUBNASALE':2,
 'MEDIAL_CANTHUS_L':362,'MEDIAL_CANTHUS_R':133,
 'LATERAL_CANTHUS_L':263,'LATERAL_CANTHUS_R':33,
 'ALARE_L':327,'ALARE_R':98,'CHEILION_L':291,'CHEILION_R':61,
 'ZYGOMA_L':454,'ZYGOMA_R':234,'MENTON':152,'GNATHION':175,
 'LABIALE_SUPERIUS':13,'LABIALE_INFERIUS':14,
 'SUPERCILIARE_L':334,'SUPERCILIARE_R':105,'FOREHEAD_PROXY':10}
PRETTY={
 'GLABELLA':'Glabella','SELLION_NASION':'Nasion','PRONASALE':'Nose tip',
 'SUBNASALE':'Below nose','MEDIAL_CANTHUS_L':'Left inner eye',
 'MEDIAL_CANTHUS_R':'Right inner eye','LATERAL_CANTHUS_L':'Left outer eye',
 'LATERAL_CANTHUS_R':'Right outer eye','ALARE_L':'Left nose wing',
 'ALARE_R':'Right nose wing','CHEILION_L':'Left mouth corner',
 'CHEILION_R':'Right mouth corner','ZYGOMA_L':'Left cheek',
 'ZYGOMA_R':'Right cheek','MENTON':'Chin','GNATHION':'Lower chin',
 'LABIALE_SUPERIUS':'Upper lip','LABIALE_INFERIUS':'Lower lip',
 'SUPERCILIARE_L':'Left superciliare (proxy)', 'SUPERCILIARE_R':'Right superciliare (proxy)',
 'FOREHEAD_PROXY':'Upper forehead (proxy)', 'TRICHION':'Trichion (manual hairline)'}
OLD_PAIRS=[('LATERAL_CANTHUS_L','LATERAL_CANTHUS_R'),('ALARE_L','ALARE_R'),
           ('CHEILION_L','CHEILION_R'),('ZYGOMA_L','ZYGOMA_R')]
NEW_PAIRS=OLD_PAIRS[1:]
ALIASES={'NOSE_TIP':'PRONASALE','EYE_L':'LATERAL_CANTHUS_L','EYE_R':'LATERAL_CANTHUS_R',
         'ALA_L':'ALARE_L','ALA_R':'ALARE_R','MOUTH_L':'CHEILION_L','MOUTH_R':'CHEILION_R'}

def finite_point(p):
    return p is not None and np.shape(p)==(3,) and bool(np.isfinite(p).all())

def with_aliases(points):
    out=dict(points);out.update({a:points[b] for a,b in ALIASES.items() if b in points});return out

# Original inherited 3D ASI function is inserted below unchanged.
def compute_metrics_3d(pts3d):
    """Pose-aware 3D horizontal + vertical facial asymmetry measurements."""
    required = {"GLABELLA","NOSE_TIP","MENTON","EYE_L","EYE_R","ALA_L","ALA_R","MOUTH_L","MOUTH_R","ZYGOMA_L","ZYGOMA_R"}
    if not pts3d or not required.issubset(pts3d):
        return None
    midline_pts=np.stack([pts3d["GLABELLA"],pts3d["NOSE_TIP"],pts3d["MENTON"]])
    p0=midline_pts.mean(axis=0)
    centered=midline_pts-p0
    try:
        _,_,vt=np.linalg.svd(centered)
    except np.linalg.LinAlgError:
        return None
    v_up=vt[0].astype(np.float32)
    if v_up[1] < 0: v_up=-v_up
    v_up=v_up/(np.linalg.norm(v_up)+1e-9)

    pairs_all=[("EYE_L","EYE_R"),("ALA_L","ALA_R"),("MOUTH_L","MOUTH_R"),("ZYGOMA_L","ZYGOMA_R")]
    pairs_upper=[("EYE_L","EYE_R"),("ZYGOMA_L","ZYGOMA_R")]
    pairs_lower=[("ALA_L","ALA_R"),("MOUTH_L","MOUTH_R")]
    lr=np.mean([pts3d[R]-pts3d[L] for L,R in pairs_all],axis=0)
    lr=lr-np.dot(lr,v_up)*v_up
    if np.linalg.norm(lr)<1e-6: return None
    n=lr/np.linalg.norm(lr)  # left-right / symmetry-plane normal
    v_forward=np.cross(n,v_up)
    if np.linalg.norm(v_forward)<1e-6: return None
    v_forward=v_forward/np.linalg.norm(v_forward)

    def region_AsI(pairs):
        vals=[]
        for L,R in pairs:
            dL=abs(float(np.dot(pts3d[L]-p0,n)))
            dR=abs(float(np.dot(pts3d[R]-p0,n)))
            vals.append(abs((dR-dL)/(dR+dL+1e-6)))
        return float(np.mean(vals)) if vals else 0.0

    vertical_mm=[]; vertical_norm=[]
    for L,R in pairs_all:
        dy=abs(float(np.dot(pts3d[L]-pts3d[R],v_up)))
        span=float(np.linalg.norm(pts3d[R]-pts3d[L]))
        vertical_mm.append(dy)
        vertical_norm.append(dy/(span+1e-6))

    AsI_upper=region_AsI(pairs_upper)
    AsI_lower=region_AsI(pairs_lower)
    AsI_overall=region_AsI(pairs_all)
    FA_vert=float(np.mean(vertical_norm)) if vertical_norm else 0.0
    Vertical_mm=float(np.mean(vertical_mm)) if vertical_mm else 0.0
    roll_deg=abs(math.degrees(math.atan2(float(v_up[0]),float(v_up[1]))))
    if max(AsI_upper,AsI_lower)<0.07: verdict="Balanced"
    elif AsI_upper>AsI_lower: verdict="Upper Asymmetry"
    else: verdict="Lower Asymmetry"
    return {"AsI_upper":AsI_upper,"AsI_lower":AsI_lower,"AsI_overall":AsI_overall,
            "FA_hor":abs(AsI_upper-AsI_lower)*2,"FA_vert":FA_vert,"Vertical_mm":Vertical_mm,
            "Angle_deg":roll_deg,"Verdict":verdict,"is_3d":True,
            "plane_point":p0,"plane_normal":n,"up_axis":v_up,"forward_axis":v_forward}

# NEW (v48): consistent experimental 3D regional ASI. Independent of the
# inherited scores above, which remain accessible for longitudinal audits.
# Matches the MATLAB bilateral-only H/V/D *definition*, not an assertion
# that independently estimated reference axes will be bit-for-bit identical.
ASI_GROUPS={
    'upper':[
        ('SUPERCILIARE_L','SUPERCILIARE_R','Brow proxies'),
        ('LATERAL_CANTHUS_L','LATERAL_CANTHUS_R','Outer canthi'),
        ('MEDIAL_CANTHUS_L','MEDIAL_CANTHUS_R','Inner canthi')],
    'lower':[
        ('ALARE_L','ALARE_R','Nose wings'),
        ('CHEILION_L','CHEILION_R','Mouth corners')]
}
# Zygoma is deliberately *not* part of the default five pairs: its left
# depth is frequently unavailable. It remains exported and QC-visible.
ASI_REFERENCE_PAIRS=[
    ('MEDIAL_CANTHUS_L','MEDIAL_CANTHUS_R'),
    ('LATERAL_CANTHUS_L','LATERAL_CANTHUS_R'),
    ('SUPERCILIARE_L','SUPERCILIARE_R')]

def landmark_tally(points, pixels=None, quality=None):
    """2D detected vs metric 3D measurements for the original 21 IDs.

    'valid_3d' says a finite depth/XYZ was returned, NOT that placement was
    anatomically verified. 'good' is the subset with strict patch status.
    """
    pixels=pixels or {};quality=quality or {}
    valid=[name for name in LM if finite_point(points.get(name))]
    detected=[name for name in LM if name in pixels and
              0<=pixels[name][0]<FRAME_SIZE[0] and 0<=pixels[name][1]<FRAME_SIZE[1]]
    good=[name for name in valid if quality.get(name,{}).get('status')=='good']
    return {'valid_3d':len(valid),'detected_2d':len(detected),
            'good_depth':len(good),'total':len(LM),
            'missing':[k for k in LM if k not in valid],
            'not_detected':[k for k in LM if k not in detected]}

def asi_reference(points):
    """A local face-oriented orthonormal basis in camera-space millimeters.

    Origin: Subnasale. V: Nasion -> Subnasale. U: mean bilateral eye/brow
    vector after removing the component parallel to V. D: U cross V.
    Changing head pose or reference landmark errors can bias all three ASIs.
    """
    if not all(finite_point(points.get(k)) for k in ('SELLION_NASION','SUBNASALE')):
        return None
    origin=np.asarray(points['SUBNASALE'],dtype=float)
    vertical=origin-np.asarray(points['SELLION_NASION'],dtype=float)
    vlength=float(np.linalg.norm(vertical))
    if vlength<2.:return None
    v=vertical/vlength
    candidates=[]
    for l,r in ASI_REFERENCE_PAIRS:
        if finite_point(points.get(l)) and finite_point(points.get(r)):
            d=np.asarray(points[r],float)-np.asarray(points[l],float)
            d=d-np.dot(d,v)*v
            norm=np.linalg.norm(d)
            if norm>5.:candidates.append(d/norm)
    if not candidates:return None
    u=np.mean(candidates,axis=0)
    unorm=float(np.linalg.norm(u))
    if unorm<0.2:return None
    u=u/unorm
    d=np.cross(u,v);dnorm=float(np.linalg.norm(d))
    if dnorm<1e-6:return None
    d=d/dnorm;u=np.cross(v,d);u=u/np.linalg.norm(u)
    return {'origin':origin,'basis':np.column_stack([u,v,d]),
            'source':'Subnasale origin; Nasion->Subnasale V; mean canthus/brow U'}

def compute_three_region_asi(points):
    """Bilateral 3D experimental ASI: 100 * sqrt(sum(deltaH)^2 +
    sum(deltaV)^2 + sum(deltaD)^2) / sum(pair_widths).

    For each corresponding left/right pair: hL=|U_L|, hR=|U_R|,
    deltaH=|hL-hR|; deltaV=|V_L-V_R|; deltaD=|D_L-D_R|;
    pair_width=hL+hR. Pairs need valid XYZ and opposite midline sides.
    Overall pools raw per-pair values instead of averaging region scores.
    No clinical cutoffs, and the normalized index may exceed 100.
    """
    ref=asi_reference(points)
    out={'reference':ref,'upper':None,'lower':None,'overall':None,'pairs':[],
         'components':{},'coverage':{},'missing_pairs':{}}
    groups={key:[] for key in ASI_GROUPS}
    for region,pairs in ASI_GROUPS.items():
        missing=[]
        for left,right,title in pairs:
            row={'region':region,'name':title,'left':left,'right':right,
                 'valid':False,'reason':'','width_mm':None,'dH_mm':None,
                 'dV_mm':None,'dD_mm':None,'pair_asi':None}
            if ref is None:row['reason']='reference unavailable'
            elif not (finite_point(points.get(left)) and finite_point(points.get(right))):
                row['reason']='missing 3D landmark'
            else:
                ql=(np.asarray(points[left],float)-ref['origin'])@ref['basis']
                qr=(np.asarray(points[right],float)-ref['origin'])@ref['basis']
                width=abs(ql[0])+abs(qr[0])
                # Both landmarks should be on opposite sides of the plane.
                if ql[0]*qr[0]>=0:row['reason']='pair not on opposite sides of plane'
                elif width<2.:row['reason']='horizontal span too small'
                else:
                    row.update(valid=True,reason='ok',width_mm=float(width),
                               dH_mm=float(abs(abs(ql[0])-abs(qr[0]))),
                               dV_mm=float(abs(ql[1]-qr[1])),
                               dD_mm=float(abs(ql[2]-qr[2])))
                    row['pair_asi']=100*math.sqrt(row['dH_mm']**2+row['dV_mm']**2+row['dD_mm']**2)/width
                    groups[region].append(row)
            if not row['valid']:missing.append(title+': '+row['reason'])
            out['pairs'].append(row)
        out['coverage'][region]=(len(groups[region]),len(pairs))
        out['missing_pairs'][region]=missing
    allpairs=groups['upper']+groups['lower']
    for region,rows in [('upper',groups['upper']),('lower',groups['lower']),('overall',allpairs)]:
        out['coverage'][region]=(len(rows),5 if region=='overall' else len(ASI_GROUPS[region]))
        if not rows:
            out['components'][region]=None
            continue
        totalW=sum(row['width_mm'] for row in rows)
        h=100*sum(row['dH_mm'] for row in rows)/totalW
        v=100*sum(row['dV_mm'] for row in rows)/totalW
        d=100*sum(row['dD_mm'] for row in rows)/totalW
        out['components'][region]={'H':h,'V':v,'D':d,'width_mm':totalW,
                                   'valid_pairs':len(rows)}
        out[region]=float(math.sqrt(h*h+v*v+d*d))
    return out


def face_reference(points):
    names=['GLABELLA','PRONASALE','MENTON','MEDIAL_CANTHUS_L','MEDIAL_CANTHUS_R']
    if any(not finite_point(points.get(k)) for k in names):return None
    origin=np.mean([points[k] for k in names[:3]],axis=0)
    down=points['MENTON']-points['GLABELLA'];length=np.linalg.norm(down)
    if length<1e-6:return None
    down=down/length
    across=points['MEDIAL_CANTHUS_R']-points['MEDIAL_CANTHUS_L'];span=float(np.linalg.norm(across))
    across=across-np.dot(across,down)*down;length=np.linalg.norm(across)
    if length<1e-6 or span<1e-6:return None
    across=across/length;depth=np.cross(across,down);depth=depth/np.linalg.norm(depth)
    return {'origin':origin,'basis':np.column_stack([across,down,depth]),'scale_mm':span,'source':'glabella/nose/menton + inner eyes'}

def metrics(points):
    # Retain the source v33/v40 old calculation's historical side assignment.
    legacy=dict(points)
    for left,right in OLD_PAIRS[1:]:
        legacy.pop(left,None);legacy.pop(right,None)
        if right in points:legacy[left]=points[right]
        if left in points:legacy[right]=points[left]
    old=compute_metrics_3d(with_aliases(legacy))
    reference=face_reference(points);rows=[]
    for left,right in OLD_PAIRS:
        row={'left':left,'right':right,'span_mm':None,'old_left_mm':None,'old_right_mm':None,
             'old_pair_pct':None,'horizontal_mm':None,'vertical_mm':None,'depth_mm':None,
             'hv_residual_mm':None,'xyz_residual_mm':None,'signed_horizontal_mm':None,
             'signed_vertical_mm':None,'signed_depth_mm':None,'reflected':None,'components':None}
        if finite_point(points.get(left)) and finite_point(points.get(right)):
            a,b=points[left],points[right];row['span_mm']=float(np.linalg.norm(b-a))
            if old:
                normal=old['plane_normal'];origin=old['plane_point']
                dl=abs(float(np.dot(a-origin,normal)));dr=abs(float(np.dot(b-origin,normal)))
                row.update(old_left_mm=dl,old_right_mm=dr,old_pair_pct=100*abs((dr-dl)/(dr+dl+1e-6)))
            if reference and (left,right) in NEW_PAIRS:
                origin,basis=reference['origin'],reference['basis']
                qa=(a-origin)@basis;qb=(b-origin)@basis;reflected=qa*np.array([-1.,1.,1.]);residual=qb-reflected
                row.update(horizontal_mm=abs(float(residual[0])),vertical_mm=abs(float(residual[1])),
                           depth_mm=abs(float(residual[2])),hv_residual_mm=float(np.linalg.norm(residual[:2])),
                           xyz_residual_mm=float(np.linalg.norm(residual)),reflected=origin+reflected@basis.T,
                           components=residual,signed_horizontal_mm=float(residual[0]),
                           signed_vertical_mm=float(residual[1]),signed_depth_mm=float(residual[2]))
        rows.append(row)
    scored=[r for r in rows[1:] if r['hv_residual_mm'] is not None]
    complete=len(scored)==3
    new={'hv_pct':None,'xyz_pct':None,'horizontal_mm':None,'vertical_mm':None,'depth_mm':None,
         'pairs_valid':len(scored),'scale_mm':reference['scale_mm'] if reference else None}
    if scored:
        for key in ['horizontal_mm','vertical_mm','depth_mm']:new[key]=float(np.mean([r[key] for r in scored]))
        if complete:
            new['hv_pct']=100*float(np.mean([r['hv_residual_mm'] for r in scored]))/reference['scale_mm']
            new['xyz_pct']=100*float(np.mean([r['xyz_residual_mm'] for r in scored]))/reference['scale_mm']
    return {'old':old,'new':new,'rows':rows,'reference':reference}


def sample_landmark_depth(depth,u,v,intr):
    """Same-frame local median; no invented Z or unconstrained hole filling."""
    if depth is None:return None,{'status':'no raw depth','source':'none'}
    h,w=depth.shape
    if not (0<=u<w and 0<=v<h):return None,{'status':'outside image','source':'none'}
    best=None
    for radius in (2,3,5):
        patch=depth[max(0,v-radius):min(h,v+radius+1),max(0,u-radius):min(w,u+radius+1)]
        values=patch[(patch>=180)&(patch<=1500)].astype(float)
        q={'valid_pixels':len(values),'patch_radius':radius,'source':'local_patch','status':'no valid depth','depth_spread_mm':None}
        if len(values)<3:best=q;continue
        median=float(np.median(values));mad=float(np.median(abs(values-median)))
        # Keep one local depth population. This does not borrow a distant facial surface.
        keep=abs(values-median)<=max(6.,3*1.4826*mad)
        values=values[keep]
        if len(values)<3:best=q;continue
        spread=float(np.percentile(values,90)-np.percentile(values,10))
        q.update(valid_pixels=len(values),depth_spread_mm=spread,median_depth_mm=float(np.median(values)))
        if spread<=40:
            q['status']='good' if radius==2 and len(values)>=5 and spread<=15 else 'caution: expanded/noisy patch'
            return deproject(u,v,float(np.median(values)),intr),q
        q['status']='unstable depth';best=q
    return None,best or {'status':'no valid depth','source':'local_patch'}

def angle_between(a,b):
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if den<1e-8 else math.degrees(math.acos(float(np.clip(np.dot(a,b)/den,-1,1))))

def bilateral_anchor(points,anchor,left,right,reference=None):
    result={'left_mm':None,'right_mm':None,'radial_asi_pct':None,'opening_angle_deg':None,
            'left_inclination_deg':None,'right_inclination_deg':None,'inclination_difference_deg':None,
            'vertical_mm':None,'horizontal_mm':None,'reference':'unavailable'}
    if any(not finite_point(points.get(k)) for k in (anchor,left,right)):return result
    a=points[left]-points[anchor];b=points[right]-points[anchor];dl=float(np.linalg.norm(a));dr=float(np.linalg.norm(b))
    result.update(left_mm=dl,right_mm=dr,radial_asi_pct=100*abs(dr-dl)/(dr+dl+1e-6),opening_angle_deg=angle_between(a,b))
    if reference:
        basis=reference['basis'];ql=a@basis;qr=b@basis
        la=angle_between(a,basis[:,1]);ra=angle_between(b,basis[:,1])
        result.update(left_inclination_deg=la,right_inclination_deg=ra,
                      inclination_difference_deg=abs(ra-la) if la is not None and ra is not None else None,
                      vertical_mm=abs(float(qr[1]-ql[1])),horizontal_mm=abs(float(qr[0]+ql[0])),
                      reference=reference.get('source','original reference'))
    return result

def extended_metrics(points):
    # Run the v40 definitions; preserve complete old/full-new scores separately.
    result=metrics(points);ref=result['reference'];new=result['new']
    core=[row for row in result['rows'][1:3] if row['hv_residual_mm'] is not None]
    new['core_hv_pct']=100*np.mean([r['hv_residual_mm'] for r in core])/ref['scale_mm'] if len(core)==2 and ref is not None else None
    new['core_xyz_pct']=100*np.mean([r['xyz_residual_mm'] for r in core])/ref['scale_mm'] if len(core)==2 and ref is not None else None
    mouth=result['rows'][2]
    new['mouth_hv_pct']=100*mouth['hv_residual_mm']/ref['scale_mm'] if ref and mouth['hv_residual_mm'] is not None else None
    result['mouth']=bilateral_anchor(points,'PRONASALE','CHEILION_L','CHEILION_R',ref)
    result['brow']=bilateral_anchor(points,'TRICHION','SUPERCILIARE_L','SUPERCILIARE_R',ref)
    result['forehead_proxy']=bilateral_anchor(points,'FOREHEAD_PROXY','SUPERCILIARE_L','SUPERCILIARE_R',ref)
    required=['GLABELLA','PRONASALE','MENTON','LATERAL_CANTHUS_L','LATERAL_CANTHUS_R','ALARE_L','ALARE_R','CHEILION_L','CHEILION_R','ZYGOMA_L','ZYGOMA_R']
    result['old_missing']=[k for k in required if not finite_point(points.get(k))]
    result['new_missing']=[k for k in ['GLABELLA','PRONASALE','MENTON','MEDIAL_CANTHUS_L','MEDIAL_CANTHUS_R',*[n for pair in NEW_PAIRS for n in pair]] if not finite_point(points.get(k))]
    result['asi']=compute_three_region_asi(points)
    return result

def burst_statistics(frames):
    keys={'Upper_3D_ASI_pct':lambda p:p.scores['asi']['upper'],
           'Lower_3D_ASI_pct':lambda p:p.scores['asi']['lower'],
           'Overall_3D_ASI_pct':lambda p:p.scores['asi']['overall'],
           'Valid_3D_Landmarks':lambda p:landmark_tally(p.landmarks)['valid_3d'],
           'Old3D_ASI_pct':lambda p:None if p.scores['old'] is None else p.scores['old']['AsI_overall']*100,
          'Full_HV_ASI_pct':lambda p:p.scores['new']['hv_pct'],
          'Core_HV_ASI_pct':lambda p:p.scores['new'].get('core_hv_pct'),
          'Pronasale_Cheilion_L_mm':lambda p:p.scores['mouth']['left_mm'],
          'Pronasale_Cheilion_R_mm':lambda p:p.scores['mouth']['right_mm'],
          'Mouth_radial_ASI_pct':lambda p:p.scores['mouth']['radial_asi_pct'],
          'Mouth_vertical_mm':lambda p:p.scores['mouth']['vertical_mm'],
          'Nose_Z_mm':lambda p:p.guide.get('distance_mm')}
    rows=[]
    for key,fn in keys.items():
        values=[fn(p) for p in frames];a=np.array([v for v in values if v is not None and np.isfinite(v)],float)
        rows.append({'metric':key,'valid_frames':len(a),'total_frames':len(frames),'mean':float(a.mean()) if len(a) else None,
                     'sd':float(a.std(ddof=1)) if len(a)>1 else None,'min':float(a.min()) if len(a) else None,'max':float(a.max()) if len(a) else None})
    return rows

OLD_BASE_METRICS=metrics

def persist_manual(packet,name,point,pixel,quality):
    packet.landmarks[name]=point;packet.landmark_pixels[name]=pixel;packet.quality[name]=quality;packet.scores=extended_metrics(packet.landmarks)
    if packet.path:
        fields=['landmark_id','x_mm','y_mm','z_mm','pixel_u','pixel_v','status','depth_spread_mm','timestamp_s']
        path=Path(packet.path)/'manual_landmarks.csv';exists=path.exists()
        with path.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields)
            if not exists:w.writeheader()
            w.writerow(dict(zip(fields,[name,*point,*pixel,quality.get('status'),quality.get('depth_spread_mm'),time.time()])))
        with (Path(packet.path)/'review_measurements.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['metric','value']);w.writerows(scalar_metrics(packet).items())


def deproject(u,v,z,intr):
    fx,fy,cx,cy=intr
    return np.array([(u-cx)*z/fx,(v-cy)*z/fy,z],dtype=np.float32)

def point_distance(points,a,b):
    if not finite_point(points.get(a)) or not finite_point(points.get(b)):return None
    return float(np.linalg.norm(points[b]-points[a]))

@dataclass
class Packet:
    seq:int
    timestamp:float
    device_meta:dict
    intrinsics:tuple
    points:np.ndarray
    colors:np.ndarray
    pixels:np.ndarray
    landmarks:dict
    landmark_pixels:dict
    quality:dict
    rgb:object=None
    guide:dict=field(default_factory=dict)
    scores:dict=field(default_factory=dict)
    path:object=None
    demo:bool=False
    pose_mode:str='Unspecified'  # Operator-selected instruction; not verified by a classifier.
    process_ms:float=0.
    depth:object=None
    frames:object=None
    selected_frame:int=0
    dense_landmarks:dict=field(default_factory=dict)
    dense_pixels:dict=field(default_factory=dict)
    expression:dict=field(default_factory=dict)


def make_guide(points,quality,pixels,w=640,h=480,asi=None):
    nose=points.get('PRONASALE');ref=face_reference(points)
    distance=float(nose[2]) if finite_point(nose) else None
    if distance is None:return {'ready':False,'distance_ok':False,'distance_mm':None,'message':'Find the face / valid nose depth','level':'warn'}
    ready=True;message='At target distance · hold still';level='good'
    if distance<TARGET_MM-TOLERANCE_MM:ready=False;message=f'Move camera BACK · target {TARGET_MM/10:.0f} cm';level='warn'
    elif distance>TARGET_MM+TOLERANCE_MM:ready=False;message=f'Move camera CLOSER · target {TARGET_MM/10:.0f} cm';level='warn'
    u,v=pixels.get('PRONASALE',(w/2,h/2))
    if abs(u-w/2)>w*.20 or abs(v-h*.43)>h*.25:ready=False;message='Center the face';level='warn'
    yaw=roll=None
    if ref:
        across=ref['basis'][:,0];yaw=math.degrees(math.atan2(abs(float(across[2])),max(1e-6,abs(float(across[0])))))
        roll=math.degrees(math.atan2(abs(float(across[1])),max(1e-6,abs(float(across[0])))))
        if yaw>15 or roll>12:ready=False;message='Look straight at camera · keep head level';level='warn'
    tally=landmark_tally(points,pixels,quality)
    missing=tally['missing']
    if asi is None:asi=compute_three_region_asi(points)
    if asi['upper'] is None or asi['lower'] is None:
        ready=False;message='Reposition: upper/lower ASI pairs unavailable';level='warn'
    elif missing and ready:
        message=f"{tally['valid_3d']}/{tally['total']} 3D landmarks; check missing points"
        level='warn'
    return {'ready':ready,'distance_ok':TARGET_MM-TOLERANCE_MM<=distance<=TARGET_MM+TOLERANCE_MM,'distance_mm':distance,'message':message,'level':level,'yaw_deg':yaw,'roll_deg':roll,'missing':missing, 'landmark_tally':tally}


def analyze_frame(rgb,depth,intr,face_mesh,seq,device_meta):
    import cv2
    started=time.perf_counter();acquired_at=time.time();h,w=rgb.shape[:2];pts={};pix={};quality={};mask=np.zeros((h,w),np.uint8)
    dense={};dense_pixels={}
    result=face_mesh.process(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
    if result.multi_face_landmarks:
        face=result.multi_face_landmarks[0].landmark
        hull=np.array([[int(np.clip(p.x*w,0,w-1)),int(np.clip(p.y*h,0,h-1))] for p in face],np.int32)
        cv2.fillConvexPoly(mask,cv2.convexHull(hull),255)
        for index,landmark in enumerate(face):
            u=int(round(landmark.x*w));v=int(round(landmark.y*h))
            if not (0<=u<w and 0<=v<h):continue
            dense_pixels[index]=(u,v)
            point,_=sample_landmark_depth(depth,u,v,intr)
            if point is not None:dense[index]=point
        for name,index in LM.items():
            landmark=face[index];u=int(round(landmark.x*w));v=int(round(landmark.y*h));pix[name]=(u,v)
            if not (0<=u<w and 0<=v<h):continue
            point,q=sample_landmark_depth(depth,u,v,intr);quality[name]=q
            if point is not None:pts[name]=point
    vv,uu=np.mgrid[0:h:CLOUD_STRIDE,0:w:CLOUD_STRIDE];zz=depth[vv,uu]
    good=(mask[vv,uu]>0)&(zz>=180)&(zz<=1500)
    u,v,z=uu[good],vv[good],zz[good].astype(np.float32)
    fx,fy,cx,cy=intr;xyz=np.column_stack([(u-cx)*z/fx,(v-cy)*z/fy,z]).astype(np.float32)
    colors=rgb[v,u,::-1].copy();pixels=np.column_stack([u,v]).astype(np.int32)
    # Dense FaceMesh pixels describe eye/eyebrow contours. Only valid stereo XYZ enters the 3D dense view.
    # Save every valid sampled point; cap only display rendering, not measurements/storage.
    packet=Packet(seq,acquired_at,dict(device_meta),tuple(intr),xyz,colors,pixels,pts,pix,quality,rgb.copy())
    packet.scores=extended_metrics(pts);packet.guide=make_guide(pts,quality,pix,w,h,packet.scores['asi'])
    packet.dense_landmarks=dense;packet.dense_pixels=dense_pixels
    packet.expression=estimate_expression(dense_pixels,guide=packet.guide)
    packet.depth=depth.copy();packet.process_ms=1000*(time.perf_counter()-started);return packet


class OakSource:
    """Synchronized RGB and RGB-aligned metric stereo depth from a real OAK.

    The Pro infrared laser pattern adds texture to the mono stereo pair. The IR
    floodlight helps mono exposures in low light; neither substitutes for RGB
    illumination required by MediaPipe FaceMesh. IR emitters are never treated
    as a new depth sensor: all XYZ values still come from stereo.depth.

    Intensities are DepthAI's normalized 0..1 settings, NOT amperes. They are
    applied after pipeline/device start (required by the host API), and we fail
    loudly if requested emitters are not available on the connected device.
    """
    def __init__(self, ir_dot=0.35, ir_flood=0.10):
        import depthai as dai
        for label, intensity in (('dot', ir_dot), ('flood', ir_flood)):
            if not np.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
                raise ValueError(f'IR {label} intensity must be between 0 and 1.')
        self.dai=dai
        self.device=None;self.pipeline=None;self.queue=None;self.intrinsics=None
        self.ir_dot_requested=float(ir_dot);self.ir_flood_requested=float(ir_flood)
        self.ir_dot=0.0;self.ir_flood=0.0;self.ir_drivers=[]
        self.name='OAK camera';self.kind='';self.platform='unknown';self.extended_disparity=False
        version=str(getattr(dai,'__version__','0'))
        self.depthai_version=version
        try:major=int(version.split('.')[0])
        except (ValueError, IndexError):raise RuntimeError('Cannot identify installed DepthAI version: '+version)
        self.v3=major>=3
        try:
            if self.v3:self._open_v3()
            else:self._open_v2()
            if (self.intrinsics is None or not np.isfinite(self.intrinsics).all()
                    or min(self.intrinsics[:2])<=0):
                raise RuntimeError('Invalid RGB camera calibration. Cannot report metric XYZ.')
        except Exception:
            try:self.close()
            except Exception:pass
            raise

    def _set_device_details(self, device):
        """Use camera metadata when available; do not infer that any OAK is Pro."""
        for method in ('getProductName','getDeviceName'):
            fn=getattr(device,method,None)
            if callable(fn):
                try:
                    name=fn()
                    if name:
                        self.name=str(name);break
                except (RuntimeError, AttributeError):pass
        try:self.platform=str(device.getPlatform()).split('.')[-1]
        except (AttributeError,RuntimeError):pass
        driver_method=getattr(device,'getIrDrivers',None)
        if callable(driver_method):
            try:self.ir_drivers=[str(driver[0]) for driver in driver_method()]
            except (AttributeError,RuntimeError,TypeError):self.ir_drivers=[]

    def _enable_ir(self):
        """Activate the Pro's emitters after streaming is started; check return values."""
        if self.ir_dot_requested<=0 and self.ir_flood_requested<=0:
            # DepthAI defaults to both IR emitters off. Avoid requiring Pro to use v59.
            return
        if self.device is None:
            raise RuntimeError('Camera not started: cannot enable infrared.')
        for label,requested,method in (
                ('dot projector',self.ir_dot_requested,'setIrLaserDotProjectorIntensity'),
                ('floodlight',self.ir_flood_requested,'setIrFloodLightIntensity')):
            if requested<=0:continue
            setter=getattr(self.device,method,None)
            if not callable(setter):
                raise RuntimeError(f'DepthAI {self.depthai_version} lacks {method}. Update depthai.')
            try:success=setter(requested)
            except Exception as exc:
                raise RuntimeError(f'Could not activate OAK-D Pro IR {label}: {exc}') from exc
            if success is False:
                raise RuntimeError(
                    f'Camera {self.name!r} refused the IR {label}. '
                    'Check that the OAK-D Pro is connected, not the Lite; '
                    'verify camera power and DepthAI firmware/library compatibility.')
            # Current DepthAI returns bool; accept None on bindings that do not
            # expose a return, but never interpret it as proof of working hardware.
            if label=='dot projector':self.ir_dot=requested
            else:self.ir_flood=requested
        print(f'Camera: {self.name} | DepthAI {self.depthai_version} ({self.kind}) '
              f'| IR dot={self.ir_dot:.2f}, flood={self.ir_flood:.2f}; '
              f'IR drivers={self.ir_drivers or "not reported"}',flush=True)

    def _intrinsics_from_rgb_calibration(self):
        w,h=FRAME_SIZE
        K=self.device.readCalibration().getCameraIntrinsics(
            self.dai.CameraBoardSocket.CAM_A,resizeWidth=w,resizeHeight=h)
        self.intrinsics=(float(K[0][0]),float(K[1][1]),float(K[0][2]),float(K[1][2]))

    def _open_v3(self):
        dai=self.dai;w,h=FRAME_SIZE
        p=dai.Pipeline();self.pipeline=p
        device=p.getDefaultDevice()
        if device is None:raise RuntimeError('No OAK camera detected. Check USB 3 cable and power.')
        self.device=device;self.kind='DepthAI v3';self._set_device_details(device)
        rgb=p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left=p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right=p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo=p.create(dai.node.StereoDepth)
        # Preserve 400P-ish close-range stereo and ~35 cm experimental target.
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        if device.getPlatform()==dai.Platform.RVC2:
            stereo.setExtendedDisparity(True);self.extended_disparity=True
        # The OAK-D Pro's 1280x800 mono sensors support native-aspect 400P.
        # Use 640x400 for close-range extended disparity, then align the depth
        # map onto the existing 640x480 RGB pixel coordinates downstream.
        left.requestOutput((640,400),fps=20.).link(stereo.left)
        right.requestOutput((640,400),fps=20.).link(stereo.right)
        color=rgb.requestOutput((w,h),type=dai.ImgFrame.Type.BGR888i,
                                resizeMode=dai.ImgResizeMode.CROP,
                                fps=20.,enableUndistortion=True)
        try:rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        except (AttributeError,RuntimeError):pass
        # Crucial: depth must use the RGB camera coordinate system before we
        # deproject MediaPipe's RGB pixel locations with RGB intrinsics.
        if device.getPlatform()==dai.Platform.RVC4:
            align=p.create(dai.node.ImageAlign)
            stereo.depth.link(align.input);color.link(align.inputAlignTo)
            depth=align.outputAligned
        else:
            color.link(stereo.inputAlignTo);depth=stereo.depth
        sync=p.create(dai.node.Sync);sync.setSyncThreshold(timedelta(milliseconds=25))
        color.link(sync.inputs['rgb']);depth.link(sync.inputs['depth'])
        self.queue=sync.out.createOutputQueue(maxSize=2,blocking=False)
        self._intrinsics_from_rgb_calibration()
        p.start()
        self._enable_ir()

    def _open_v2(self):
        dai=self.dai;w,h=FRAME_SIZE;p=dai.Pipeline();self.pipeline=p
        rgb=p.create(dai.node.ColorCamera);rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        rgb.setPreviewSize(w,h);rgb.setPreviewKeepAspectRatio(False)
        rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb.setInterleaved(True);rgb.setFps(20)
        left=p.create(dai.node.MonoCamera);right=p.create(dai.node.MonoCamera)
        for node,socket in ((left,dai.CameraBoardSocket.CAM_B),(right,dai.CameraBoardSocket.CAM_C)):
            node.setBoardSocket(socket)
            node.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            node.setFps(20)
        stereo=p.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True);stereo.setExtendedDisparity(True)
        self.extended_disparity=True
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A);stereo.setOutputSize(w,h)
        left.out.link(stereo.left);right.out.link(stereo.right)
        if not hasattr(dai.node,'Sync'):
            raise RuntimeError('This DepthAI v2 has no Sync node. Use DepthAI v3 or recent v2.')
        sync=p.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=25))
        rgb.preview.link(sync.inputs['rgb']);stereo.depth.link(sync.inputs['depth'])
        output=p.create(dai.node.XLinkOut)
        output.setStreamName('sync');sync.out.link(output.input)
        self.device=dai.Device(p)
        self.kind='DepthAI v2';self._set_device_details(self.device)
        self.queue=self.device.getOutputQueue('sync',maxSize=2,blocking=False)
        self._intrinsics_from_rgb_calibration()
        self._enable_ir()

    def read(self):
        group=self.queue.tryGet()
        if group is None:return None
        r,d=group['rgb'],group['depth']
        rgb,depth=r.getCvFrame(),d.getFrame()
        if rgb.shape[:2]!=depth.shape:
            raise RuntimeError(f'RGB/depth sizes differ: {rgb.shape[:2]} vs {depth.shape}. '
                               'Cannot map face landmarks to metric XYZ.')
        if rgb.shape[:2]!=(FRAME_SIZE[1],FRAME_SIZE[0]) or rgb.ndim!=3 or rgb.shape[-1]!=3:
            raise RuntimeError(f'Unexpected RGB frame shape {rgb.shape}; expected 480x640 BGR.')
        if depth.dtype!=np.uint16:
            raise RuntimeError(f'Expected uint16 stereo depth in millimeters; got {depth.dtype}.')
        meta={'rgb_sequence':int(r.getSequenceNum()),
              'depth_sequence':int(d.getSequenceNum()),
              'rgb_device_time_s':float(r.getTimestampDevice().total_seconds()),
              'depth_device_time_s':float(d.getTimestampDevice().total_seconds()),
              'camera_model':self.name,'camera_platform':self.platform,
              'depthai_version':self.depthai_version,
              'depth_unit':'mm',
              'ir_dot_intensity':self.ir_dot,'ir_flood_intensity':self.ir_flood,
              'ir_drivers':','.join(self.ir_drivers),
              'rgb_depth_aligned':1,'stereo_extended_disparity':int(self.extended_disparity)}
        meta['sync_delta_ms']=1000*abs(meta['rgb_device_time_s']-meta['depth_device_time_s'])
        return rgb,depth,meta

    def close(self):
        # Extinguish the IR emitters even if camera startup failed halfway.
        if self.device is not None:
            for method in ('setIrLaserDotProjectorIntensity','setIrFloodLightIntensity'):
                try:
                    setter=getattr(self.device,method,None)
                    if callable(setter):setter(0.0)
                except Exception:pass
        if self.pipeline is not None and self.v3:
            self.pipeline.stop()
        if self.device is not None and not self.v3:
            self.device.close()


def demo_packet(seq=0):
    """Explicit artificial surface for UI demonstration/testing; not camera data."""
    import cv2
    xx,yy=np.meshgrid(np.linspace(-78,78,220),np.linspace(-110,105,280));inside=(xx/78)**2+(yy/110)**2<1
    x,y=xx[inside],yy[inside];z=340-25*np.exp(-(x/24)**2-(y/38)**2)+.0015*(x*x+y*y)
    xyz=np.column_stack([x,y,z]).astype(np.float32);xyz[:,2]+=3*math.sin(seq/12)
    pts={'GLABELLA':np.array([0.,-65,340]),'PRONASALE':np.array([0.,-8,315]),'MENTON':np.array([0.,95,350]),
         'MEDIAL_CANTHUS_L':np.array([-18.,-44,335]),'MEDIAL_CANTHUS_R':np.array([18.,-44,335]),
         'LATERAL_CANTHUS_L':np.array([-43.,-44,345]),'LATERAL_CANTHUS_R':np.array([43.,-44,345]),
         'ALARE_L':np.array([-16.,0,323]),'ALARE_R':np.array([16.,0,323]),
         'CHEILION_L':np.array([-26.,43,340]),'CHEILION_R':np.array([23.,49+3*math.sin(seq/10),340]),
         'ZYGOMA_L':np.array([-66.,-9,360]),'ZYGOMA_R':np.array([66.,-9,360]),
         'SELLION_NASION':np.array([0.,-45,330]),'SUBNASALE':np.array([0.,12,330]),'SUPERCILIARE_L':np.array([-35.,-65,342]),'SUPERCILIARE_R':np.array([35.,-65,342]),'FOREHEAD_PROXY':np.array([0.,-100,349])}
    xyz[:,0]*=-1
    for p in pts.values():p[0]*=-1;p[2]+=3*math.sin(seq/12)
    intr=(462.,462.,320.,240.);pixels=np.column_stack([320+xyz[:,0]*462/xyz[:,2],240+xyz[:,1]*462/xyz[:,2]]).astype(np.int32)
    colors=np.tile([166,191,201],(len(xyz),1)).astype(np.uint8)
    rgb=np.full((480,640,3),(24,17,12),np.uint8);good=(pixels[:,0]>=0)&(pixels[:,0]<640)&(pixels[:,1]>=0)&(pixels[:,1]<480)
    rgb[pixels[good,1],pixels[good,0]]=colors[good,::-1]
    pix={k:(int(320+p[0]*462/p[2]),int(240+p[1]*462/p[2])) for k,p in pts.items()}
    packet=Packet(seq,time.time(),{'depth_sequence':seq,'rgb_sequence':seq},intr,xyz,colors,pixels,pts,pix,{},rgb,demo=True)
    packet.scores=extended_metrics(pts);packet.guide={'ready':True,'distance_mm':float(pts['PRONASALE'][2]),'message':'DEMO · synthetic data only','level':'warn','landmark_tally':landmark_tally(pts,pix,{})}
    return packet

class Engine(threading.Thread):
    def __init__(self,demo=False,ir_dot=0.35,ir_flood=0.10):
        super().__init__(daemon=True);self.demo=demo;self.ir_dot=ir_dot;self.ir_flood=ir_flood;self.active=threading.Event();self.stop_event=threading.Event()
        self.lock=threading.Lock();self.latest=None;self.history=deque(maxlen=8);self.metric_history=deque(maxlen=SMOOTHING_MAX_N);self.error=None;self.state='Starting camera…';self.fps=0.;self.source=None;self.expression_baseline={};self.calibration_samples=None
    def calibrate_expression(self):
        with self.lock:self.expression_baseline={};self.calibration_samples=[]
    def snapshot(self):
        with self.lock:return self.latest
    def run(self):
        mesh=None
        try:
            if not self.demo:
                import mediapipe as mp
                if not hasattr(mp,'solutions'):raise RuntimeError('MediaPipe FaceMesh API missing. Use your existing mediapipe==0.10.21 environment.')
                self.source=OakSource(ir_dot=self.ir_dot,ir_flood=self.ir_flood)
                mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=False,min_detection_confidence=.6,min_tracking_confidence=.6)
                self.state=(f'{self.source.name} connected · {self.source.kind} · '
                             f'IR dot {self.source.ir_dot:.2f} / flood {self.source.ir_flood:.2f}')
            else:self.state='DEMO · no real camera measurements'
            previous=-1;seq=0;times=deque(maxlen=20)
            while not self.stop_event.is_set():
                if not self.active.wait(.1):continue
                if self.demo:packet=demo_packet(seq);time.sleep(.07)
                else:
                    item=self.source.read()
                    if item is None:time.sleep(.004);continue
                    rgb,depth,meta=item
                    if meta['depth_sequence']==previous:continue
                    previous=meta['depth_sequence']
                    packet=analyze_frame(rgb,depth,self.source.intrinsics,mesh,seq,meta)
                packet.device_meta['monotonic_s']=time.monotonic()
                seq+=1;times.append(time.monotonic())
                if len(times)>1:self.fps=(len(times)-1)/(times[-1]-times[0])
                with self.lock:
                    packet.expression=estimate_expression(packet.dense_pixels,self.expression_baseline,packet.guide)
                    if self.calibration_samples is not None:
                        f=packet.expression['features']
                        if f and min(f['eye_L'],f['eye_R'])>.16 and not packet.expression['label'].startswith('Uncertain'):
                            self.calibration_samples.append(f)
                        if len(self.calibration_samples)>=15:
                            self.expression_baseline={key:float(np.median([f[key] for f in self.calibration_samples])) for key in self.calibration_samples[0]}
                            self.calibration_samples=None
                            packet.expression=estimate_expression(packet.dense_pixels,self.expression_baseline,packet.guide)
                        else:packet.expression['label']=f'Calibrating relaxed face: {len(self.calibration_samples)} of 15'
                    if self.latest is not None and packet.timestamp-self.latest.timestamp>STALE_SECONDS:
                        self.metric_history.clear()
                    self.latest=packet;self.history.append(packet)
                    from dataclasses import replace
                    self.metric_history.append(replace(packet,points=np.empty((0,3)),colors=np.empty((0,3)),pixels=np.empty((0,2)),rgb=None,depth=None,dense_landmarks={},dense_pixels={}))
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}';self.state='Camera unavailable';traceback.print_exc()
        finally:
            if mesh is not None:mesh.close()
            if self.source is not None:
                try:self.source.close()
                except Exception:traceback.print_exc()
    def stop(self):self.stop_event.set();self.active.set()


def packet_age(packet):
    monotonic=packet.device_meta.get('monotonic_s')
    return max(0,time.monotonic()-float(monotonic)) if monotonic is not None else max(0,time.time()-packet.timestamp)

def safe_id(text):
    import re
    return re.sub(r'[^A-Za-z0-9_-]','_',text.strip())[:40] or 'Anonymous'

def scalar_metrics(packet):
    m=packet.scores;out={};old=m.get('old');new=m['new'];asi=m['asi']
    tally=landmark_tally(packet.landmarks,packet.landmark_pixels,packet.quality)
    out.update({'Upper_3D_ASI_pct':asi['upper'],'Lower_3D_ASI_pct':asi['lower'],
                'Overall_3D_ASI_pct':asi['overall'],
                'Upper_ASI_pairs_valid':asi['coverage']['upper'][0],
                'Lower_ASI_pairs_valid':asi['coverage']['lower'][0],
                'Overall_ASI_pairs_valid':asi['coverage']['overall'][0],
                'Detected_2D_landmarks':tally['detected_2d'],
                'Valid_3D_landmarks':tally['valid_3d'],
                'Good_depth_landmarks':tally['good_depth'],
                'Auto_landmark_ids_total':tally['total'],
                'Selected_pose_mode':packet.pose_mode})
    for k in ['AsI_overall','AsI_upper','AsI_lower']:
        out['Old3D_'+k+'_pct']=None if old is None else 100*old[k]
    out.update({'New_HV_ASI_pct':new['hv_pct'],'New_XYZ_ASI_pct':new['xyz_pct'],
                'New_horizontal_mm':new['horizontal_mm'],'New_vertical_mm':new['vertical_mm'],
                'New_depth_mm':new['depth_mm'],'New_pairs_valid':new['pairs_valid'],
                'Reference_span_mm':new['scale_mm'],'Nose_depth_mm':packet.guide.get('distance_mm')})
    for category in ('mouth','brow','forehead_proxy'):
        for key,value in m.get(category,{}).items():
            if isinstance(value,(int,float)) or value is None:out[category+'_'+key]=value
    out['Core_HV_ASI_pct']=new.get('core_hv_pct');out['Mouth_HV_ASI_pct']=new.get('mouth_hv_pct')
    return out

SETTINGS_PATH=Path(os.environ.get('FACEASI_SETTINGS_FILE',str(Path.home()/'.config'/'faceasi'/'settings.json')))

def read_settings():
    import json
    try:
        value=json.loads(SETTINGS_PATH.read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}

def save_settings(**updates):
    import json,tempfile
    SETTINGS_PATH.parent.mkdir(parents=True,exist_ok=True)
    value=read_settings();value.update(updates)
    fd,name=tempfile.mkstemp(dir=SETTINGS_PATH.parent,prefix='.settings-')
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
        os.replace(name,SETTINGS_PATH)
    finally:
        if os.path.exists(name):os.unlink(name)

def export_folder_candidates():
    """Bounded search of existing local sync folders; never invent a Dropbox root."""
    choices=[]
    for value in (os.environ.get('FACEASI_SCANS_DIR'),read_settings().get('scans_dir')):
        if value:
            path=Path(value).expanduser()
            if path.is_absolute() and path.is_dir():choices.append(path)
    home=Path.home()
    roots=[home/name for name in ('DropboxSync','dropboxsync','Dropbox','dropbox','Dropbox Sync')]
    try:roots.extend(p for p in home.iterdir() if p.is_dir() and 'dropbox' in p.name.casefold())
    except OSError:pass
    visited=set();budget=400
    for root in roots:
        if not root.is_dir():continue
        pending=[(root,0)]
        while pending and budget>0:
            parent,depth=pending.pop(0)
            try:
                key=parent.resolve()
                if key in visited:continue
                visited.add(key);budget-=1
                if parent.name.casefold() in ('scans','results'):choices.append(parent)
                if depth>=3:continue
                pending.extend((p,depth+1) for p in sorted(parent.iterdir())
                    if p.is_dir() and not p.is_symlink() and not p.name.startswith('.'))
            except OSError:continue
    return list({str(p.resolve()):p.resolve() for p in choices}.values())


def configured_scans_dir():
    # A stale environment path must not hide a valid saved destination.
    for value in (os.environ.get('FACEASI_SCANS_DIR'),read_settings().get('scans_dir')):
        if value:
            path=Path(value).expanduser()
            if path.is_absolute() and path.is_dir():return path
    choices=export_folder_candidates()
    return choices[0] if len(choices)==1 else None


def remember_export_folder(destination,capture_folder=None):
    """Check the real destination before saving it or starting an export."""
    import tempfile
    target=Path(destination).expanduser()
    if not target.is_absolute() or not target.is_dir():
        raise ValueError('Select an existing folder, using its full path.')
    target=target.resolve()
    if capture_folder and target.is_relative_to(Path(capture_folder).resolve()):
        raise ValueError("Select a folder outside this scan's internal capture folder.")
    fd,name=tempfile.mkstemp(prefix='.faceasi-write-check-',dir=target)
    os.close(fd);Path(name).unlink()
    save_settings(scans_dir=str(target))
    # Keep an old launch-time override from reversing this selection in this process.
    if 'FACEASI_SCANS_DIR' in os.environ:os.environ['FACEASI_SCANS_DIR']=str(target)
    return target


def reserve_scan_id():
    """SQLite serializes concurrent reservations; durable across restarts and bad RTC.
    IDs have at least three digits; grow to 1000 rather than wrap/overwrite.
    """
    import sqlite3,re
    DATA_ROOT.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(str(DATA_ROOT.parent/'scan_sequence.sqlite3'),timeout=15)
    try:
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('CREATE TABLE IF NOT EXISTS sequence (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL)')
        connection.execute('BEGIN IMMEDIATE')
        current=connection.execute('SELECT value FROM sequence WHERE id=1').fetchone()
        highest=current[0] if current else 0
        # Recover from a lost/replaced database using saved scans and sync-folder files.
        for folder,pattern in ((DATA_ROOT,'*'),(configured_scans_dir(),'*.csv')):
            if folder:
                for path in folder.glob(pattern):
                    match=re.match(r'^(?:Scan_)?(\d+)_',path.name)
                    if match:highest=max(highest,int(match.group(1)))
        value=highest+1
        connection.execute('INSERT OR REPLACE INTO sequence(id,value) VALUES(1,?)',(value,));connection.commit()
        return value
    finally:connection.close()

def compact_capture_frame(packet,pose,max_points=8000):
    from dataclasses import replace
    step=max(1,math.ceil(len(packet.points)/max_points));index=slice(None,None,step)
    # Keep labeled XYZ exact; cloud thinning affects visualization/storage only.
    return replace(packet,pose_mode=pose,points=packet.points[index].copy(),colors=packet.colors[index].copy(),
                   pixels=packet.pixels[index].copy(),device_meta={**packet.device_meta,
                   'monotonic_s':packet.device_meta.get('monotonic_s',time.monotonic()),
                   'cloud_sampling':f'stored cloud stride {step}; measured landmarks unchanged'},
                   rgb=packet.rgb.copy() if packet.rgb is not None else None,
                   depth=packet.depth.copy() if packet.depth is not None else None)

def _finite_score(value):
    return value is not None and isinstance(value,(int,float,np.integer,np.floating)) and math.isfinite(float(value))


def _asi_signature(asi):
    """Window comparability: only pool identical valid pair subsets.

    Otherwise score changes can be created by losing a landmark rather than
    facial motion. The returned smoothed score remains an experimental index.
    """
    return tuple((row['left'],row['right']) for row in asi['pairs'] if row['valid'])


def compute_smoothed_asi(history, n=5):
    """CAUSAL rolling N-frame mean + median, newest observation every frame.

    All input frames remain raw. Missing current scores remain missing.
    Only previous frames with the SAME valid bilateral pair signature are used.
    Previous samples with different coverage are excluded, not set to zero.
    """
    if not history:return None
    n=max(1,min(SMOOTHING_MAX_N,int(n)))
    current=history[-1]
    asi=current.scores['asi']
    sig=_asi_signature(asi)
    ans={'window_n':n,'mean':{},'median':{},'raw':{},'n_used':{},'pair_signature':sig}
    for group in ('upper','lower','overall'):
        raw=asi[group];ans['raw'][group]=raw
        if not _finite_score(raw):
            ans['mean'][group]=None;ans['median'][group]=None;ans['n_used'][group]=0
            continue
        current_region={r['name'] for r in asi['pairs'] if r['valid'] and (group=='overall' or r['region']==group)}
        series=[]
        for frame in list(history)[-n:]:
            src=frame.scores['asi']
            relevant={r['name'] for r in src['pairs'] if r['valid'] and (group=='overall' or r['region']==group)}
            value=src[group]
            if relevant==current_region and _finite_score(value):series.append(float(value))
        ans['mean'][group]=float(np.mean(series)) if series else None
        ans['median'][group]=float(np.median(series)) if series else None
        ans['n_used'][group]=len(series)
    return ans


def asi_display_summary(frames,window=5,enabled=True,method='mean'):
    """One source for primary ASI numbers; only past/current frames are used."""
    if method not in ('mean','median'):raise ValueError('Filter must be mean or median')
    result=compute_smoothed_asi(frames,window if enabled else 1)
    if result is None:return None
    result['display']=result[method] if enabled else result['raw']
    result['label']=(f"Moving {'average' if method=='mean' else 'median'} N={result['window_n']}" if enabled else 'Raw / filter off')
    return result


# 2D diagnostic geometry, separate from stereo-depth ASI.
# Eye/brow index sets follow the official MediaPipe face_mesh_connections.py.
EXPRESSION_METHOD='2D landmark heuristic v1; unvalidated thresholds; not an ASI quality test'

def expression_features(pixels):
    required=(33,133,160,144,158,153,362,263,385,380,387,373,105,334,61,291,13,14,2)
    if any(k not in pixels for k in required):return {}
    p={k:np.asarray(pixels[k],float) for k in required}
    if any(np.shape(q)!=(2,) or not np.isfinite(q).all() for q in p.values()):return {}
    across=p[263]-p[33];span=float(np.linalg.norm(across))
    if span<20:return {}
    across/=span;down=np.array([-across[1],across[0]])
    if np.dot(p[2]-(p[33]+p[263])/2,down)<0:down=-down
    result={}
    for side,a,b,u1,d1,u2,d2,brow in [('L',362,263,385,380,387,373,334),('R',33,133,160,144,158,153,105)]:
        width=float(np.linalg.norm(p[b]-p[a]))
        if width<6:return {}
        result['eye_'+side]=float((np.linalg.norm(p[u1]-p[d1])+np.linalg.norm(p[u2]-p[d2]))/(2*width))
        result['brow_'+side]=float(np.dot((p[a]+p[b])/2-p[brow],down)/width)
    center=(p[13]+p[14])/2;corners=(p[61]+p[291])/2
    result['mouth_width']=float(np.linalg.norm(p[291]-p[61])/span)
    result['mouth_open']=float(np.linalg.norm(p[13]-p[14])/span)
    result['mouth_lift']=float(np.dot(center-corners,down)/span)
    result['mouth_tilt']=float(abs(np.dot(p[291]-p[61],down))/span)
    for kind in ('eye','brow'):
        l,r=result[kind+'_L'],result[kind+'_R']
        result[kind+'_difference']=100*abs(l-r)/max(abs(l)+abs(r),.04)
    return result


def estimate_expression(pixels,baseline=None,guide=None):
    f=expression_features(pixels);baseline=dict(baseline or {})
    out={'label':'Uncertain / face points missing','features':f,'baseline':baseline,'calibrated':bool(baseline),'method':EXPRESSION_METHOD,'note':''}
    if not f:return out
    if guide and ((guide.get('yaw_deg') or 0)>20 or (guide.get('roll_deg') or 0)>18):
        out['label']='Uncertain / head turned';return out
    if not .55<f['eye_L']/max(f['eye_R'],.01)<1.8 and min(f['eye_L'],f['eye_R'])>.25:
        out['label']='Uncertain / check eye points';return out
    labels=[]
    closed={side:f['eye_'+side] < max(.10,.55*baseline.get('eye_'+side,.29)) for side in ('L','R')}
    if closed['L'] and closed['R']:labels.append('Eyes closed / blinking')
    elif closed['L'] or closed['R']:labels.append('Wink / asymmetric eyes')
    smiling=(f['mouth_width']>baseline.get('mouth_width',.40)*1.12 and f['mouth_lift']>baseline.get('mouth_lift',.01)+.015)
    if smiling:labels.append('Smiling-like')
    elif f['mouth_open']>max(.12,baseline.get('mouth_open',.02)+.10):labels.append('Mouth open / other expression')
    if baseline:
        dl=f['brow_L']-baseline.get('brow_L',f['brow_L']);dr=f['brow_R']-baseline.get('brow_R',f['brow_R'])
        if abs(dl-dr)>.18:labels.append('Asymmetric / distorted expression')
        elif (dl+dr)/2>.15:labels.append('Eyebrows raised')
        elif f['mouth_tilt']>baseline.get('mouth_tilt',0)+.10:labels.append('Asymmetric / distorted expression')
    elif f['brow_difference']>35 or f['mouth_tilt']>.15:labels.append('Asymmetric expression')
    out['label']=' + '.join(labels) if labels else ('Neutral-like' if baseline else 'Neutral / unsure')
    out['note']='Person-specific neutral baseline' if baseline else 'Uncalibrated: calibrate with relaxed face and eyes open'
    return out


def expression_line(packet):
    e=packet.expression
    return 'Expression estimate: '+e.get('label','Unavailable')+(' [uncalibrated]' if e.get('features') and not e.get('calibrated') else '')


def save_expression(packet,path):
    e=packet.expression
    with Path(path).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['key','value'])
        for k in ('label','method','note'):w.writerow([k,e.get(k,'')])
        w.writerow(['calibrated',int(e.get('calibrated',False))])
        for section in ('features','baseline'):
            for k,value in e.get(section,{}).items():w.writerow([section+'.'+k,value])


def load_expression(packet,path):
    if not Path(path).is_file():
        packet.expression=estimate_expression(packet.dense_pixels,guide=packet.guide);return
    with Path(path).open(newline='') as f:rows={r['key']:r['value'] for r in csv.DictReader(f)}
    packet.expression={k:rows.get(k,'') for k in ('label','method','note')}
    packet.expression['calibrated']=rows.get('calibrated')=='1'
    for section in ('features','baseline'):
        packet.expression[section]={k.split('.',1)[1]:float(v) for k,v in rows.items() if k.startswith(section+'.') and v!=''}


def scan_asi_summary(frames):
    """Mean raw frame ASIs, using one consistent pair subset per region.

    Most frequent valid subset wins (tie: more pairs). Each included frame has
    equal weight. This is a recording summary, independent of playback window.
    """
    result={'values':{},'used':{},'total':len(frames),'pairs':{}}
    for region in ('upper','lower','overall'):
        groups={}
        for frame in frames:
            asi=frame.scores['asi'];value=asi[region]
            if not _finite_score(value):continue
            signature=tuple(sorted(row['name'] for row in asi['pairs'] if row['valid'] and (region=='overall' or row['region']==region)))
            if signature:groups.setdefault(signature,[]).append(float(value))
        signature=max(groups,key=lambda k:(len(groups[k]),len(k),k)) if groups else ()
        values=groups.get(signature,[])
        result['values'][region]=float(np.mean(values)) if values else None
        result['used'][region]=len(values);result['pairs'][region]=signature
    return result


def landmark_overlay(image,packet):
    import cv2
    image=image.copy()
    for name,(u,v) in packet.landmark_pixels.items():
        cv2.circle(image,(int(u),int(v)),3,(60,225,235) if name in packet.landmarks else (110,110,110),-1)
    return image


def dense_overlay(image,packet):
    """2D detected pixels remain visible even where stereo depth is missing."""
    import cv2
    image=image.copy()
    for index,(u,v) in packet.dense_pixels.items():
        cv2.circle(image,(int(u),int(v)),1,(70,225,240) if index in packet.dense_landmarks else (120,120,120),-1)
    return image


def save_dense_landmarks(packet,path):
    with Path(path).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['landmark_id','x_mm','y_mm','z_mm','pixel_u','pixel_v'])
        for index,uv in sorted(packet.dense_pixels.items()):
            point=packet.dense_landmarks.get(index)
            writer.writerow([index,*(point if finite_point(point) else ('','','')),*uv])


def load_dense_landmarks(packet,path):
    if not Path(path).is_file():return
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            index=int(row['landmark_id']);packet.dense_pixels[index]=(int(row['pixel_u']),int(row['pixel_v']))
            if all(row[k] for k in ('x_mm','y_mm','z_mm')):
                point=np.array([float(row[k]) for k in ('x_mm','y_mm','z_mm')],np.float32)
                if finite_point(point):packet.dense_landmarks[index]=point


def smooth_value(frames,getter,window=5,method='mean'):
    """Causal scalar filter; an invalid current value stays invalid, never imputed."""
    if not frames:return None
    current=getter(frames[-1])
    if current is None or not np.isfinite(current):return None
    values=[getter(f) for f in frames[-max(1,int(window)):]]
    values=[v for v in values if v is not None and np.isfinite(v)]
    if not values:return None
    return float(np.median(values) if method=='median' else np.mean(values))

def export_matlab_csv(folder,destination,progress=None,filter_window=None,filter_method=None):
    """One atomic long-format CSV: all labeled XYZ, ASI and cloud frames.
    Export uses saved raw geometry, never smoothed/fabricated XYZ. No photo files.
    """
    import tempfile,hashlib,json
    def encoded(value):
        def clean(x):
            if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
            if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
            if isinstance(x,np.generic):x=x.item()
            if isinstance(x,float) and not math.isfinite(x):return None
            return x
        return json.dumps(clean(value),separators=(',',':'),allow_nan=False,default=str)
    source=Path(folder);target=Path(destination).expanduser()
    if target.resolve().is_relative_to(source.resolve()):raise ValueError('Choose a scans folder outside the local capture folder.')
    if not target.is_dir():raise ValueError('The selected export folder does not exist. Choose an existing folder.')
    with (source/'metadata.csv').open(newline='') as f:meta={r['key']:r['value'] for r in csv.DictReader(f)}
    scan=meta.get('scan_id')
    if not scan:
        scan=f'{reserve_scan_id():03d}';meta['scan_id']=scan
        temporary=source/'.metadata.partial'
        with temporary.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['key','value']);w.writerows(meta.items());f.flush();os.fsync(f.fileno())
        os.replace(temporary,source/'metadata.csv')
    mode={'Neutral':'N','Normal':'N','Smiling':'S','Blinking':'B'}.get(meta.get('pose_mode'))
    if mode is None:raise ValueError('Select a known Normal, Smiling or Blinking expression before exporting.')
    pose_name={'N':'Neutral','S':'Smile','B':'Blink'}[mode]
    name=f'Scan_{int(scan):03d}_{pose_name}'+('_DEMO' if meta.get('demo')=='1' else '')+'.csv'
    final=target/name;scan_key=source.name
    if final.exists():
        with final.open(newline='') as f:existing=next(csv.DictReader(f),{})
        if existing.get('scan_key')!=scan_key:raise FileExistsError('This scan ID already exists for a different recording. Choose another scans folder; existing CSV was not overwritten.')
    manifest=[]
    if (source/'burst_manifest.csv').exists():
        with (source/'burst_manifest.csv').open(newline='') as f:manifest=list(csv.DictReader(f))
    folders=[source/r['folder'] for r in manifest] if manifest else [source]
    fields=['schema_version','scan_id','subject_id','mode','scan_key','frame_index','elapsed_s','record_type','landmark_id',
            'x_mm','y_mm','z_mm','r','g','b','pixel_u','pixel_v','valid_depth_pixels','depth_spread_mm',
            'upper_asi_raw_pct','lower_asi_raw_pct','overall_asi_raw_pct',
            'upper_asi_filtered_pct','lower_asi_filtered_pct','overall_asi_filtered_pct',
            'filter_window','filter_method','frame_count','wall_clock_verified','demo',
            'upper_pairs','lower_pairs','overall_pairs','upper_filter_samples','lower_filter_samples','overall_filter_samples','scan_upper_mean_pct','scan_lower_mean_pct','scan_overall_mean_pct','summary_method','expression_estimate','expression_calibrated','eye_gap_ratio_L','eye_gap_ratio_R','brow_gap_ratio_L','brow_gap_ratio_R','eye_opening_asymmetry_2d_pct','brow_height_asymmetry_2d_pct','metrics_json','quality_json','device_metadata_json','expression_json','intrinsics_json','timestamp','old_upper_asi_pct','old_lower_asi_pct','old_overall_asi_pct','new_hv_asi_pct','new_xyz_asi_pct','core_hv_asi_pct','upper_pair_subset','lower_pair_subset','overall_pair_subset']
    window=max(1,min(SMOOTHING_MAX_N,int(filter_window if filter_window is not None else float(meta.get('smooth_window',5)))))
    method=filter_method or meta.get('smooth_method','mean')
    if method not in ('mean','median'):raise ValueError('Filter method must be mean or median')
    fd,temp=tempfile.mkstemp(dir=target,prefix='.'+name+'.',suffix='.partial');history=[];summary_frames=[]
    try:
        with os.fdopen(fd,'w',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
            for i,path in enumerate(folders,1):
                frame=load_packet(path);history.append(frame)
                # Retain only the selected window of lightweight history packets for scalar smoothing.
                from dataclasses import replace
                history[-1]=replace(frame,points=np.empty((0,3)),colors=np.empty((0,3)),pixels=np.empty((0,2)),rgb=None,depth=None)
                summary_frames.append(history[-1])
                history=history[-max(window,1):]
                raw=frame.scores['asi']
                expression=frame.expression;features=expression.get('features',{})
                base={'schema_version':'faceasi.scan.v1','scan_id':f'{int(scan):03d}',
                      'subject_id':meta.get('subject_id',meta.get('patient_id','Unknown')),
                      'mode':{'N':'Normal','S':'Smiling','B':'Blinking'}[mode],'scan_key':scan_key,
                      'frame_index':i,'elapsed_s':frame.device_meta.get('elapsed_s',''),
                      'frame_count':len(folders),'wall_clock_verified':0,'demo':meta.get('demo',0),
                      'filter_window':window,'filter_method':method,
                      'expression_estimate':expression.get('label','Unavailable'),'expression_calibrated':int(expression.get('calibrated',False)),
                      **{'eye_gap_ratio_'+side:features.get('eye_'+side) for side in ('L','R')},
                      **{'brow_gap_ratio_'+side:features.get('brow_'+side) for side in ('L','R')},
                      'eye_opening_asymmetry_2d_pct':features.get('eye_difference'),
                      'brow_height_asymmetry_2d_pct':features.get('brow_difference')}
                scalars=scalar_metrics(frame)
                base.update({
                    'old_upper_asi_pct':scalars['Old3D_AsI_upper_pct'],
                    'old_lower_asi_pct':scalars['Old3D_AsI_lower_pct'],
                    'old_overall_asi_pct':scalars['Old3D_AsI_overall_pct'],
                    'new_hv_asi_pct':scalars['New_HV_ASI_pct'],
                    'new_xyz_asi_pct':scalars['New_XYZ_ASI_pct'],
                    'core_hv_asi_pct':scalars['Core_HV_ASI_pct']})
                filtered=asi_display_summary(history,window,True,method)
                for region in ('upper','lower','overall'):
                    base[region+'_filter_samples']=filtered['n_used'][region]
                    base[region+'_asi_raw_pct']=raw[region]
                    base[region+'_asi_filtered_pct']=filtered['display'][region]
                    base[region+'_pairs']=raw['coverage'][region][0]
                    base[region+'_pair_subset']=encoded(sorted(row['name'] for row in raw['pairs'] if row['valid'] and (region=='overall' or row['region']==region)))
                writer.writerow({**base,'record_type':'frame','timestamp':frame.timestamp,
                    'metrics_json':encoded(frame.scores),'quality_json':encoded(frame.quality),
                    'device_metadata_json':encoded(frame.device_meta),'expression_json':encoded(frame.expression),
                    'intrinsics_json':encoded(frame.intrinsics)})
                for landmark in list(LM)+(['TRICHION'] if 'TRICHION' in frame.landmarks else []):
                    xyz=frame.landmarks.get(landmark);uv=frame.landmark_pixels.get(landmark,('',''));quality=frame.quality.get(landmark,{})
                    writer.writerow({**base,'record_type':'landmark','landmark_id':landmark,
                                     **dict(zip(('x_mm','y_mm','z_mm'),xyz if finite_point(xyz) else ('','',''))),
                                     'pixel_u':uv[0],'pixel_v':uv[1],'valid_depth_pixels':quality.get('valid_pixels',''),
                                     'depth_spread_mm':quality.get('depth_spread_mm','')})
                for index,uv in sorted(frame.dense_pixels.items()):
                    xyz=frame.dense_landmarks.get(index)
                    writer.writerow({'record_type':'dense_landmark','frame_index':i,'landmark_id':index,
                                     **dict(zip(('x_mm','y_mm','z_mm'),xyz if finite_point(xyz) else ('','',''))),
                                     'pixel_u':uv[0],'pixel_v':uv[1]})
                # Cloud rows omit repeated metadata; frame_index is the join key.
                for xyz,rgb,uv in zip(frame.points,frame.colors,frame.pixels):
                    writer.writerow({'record_type':'cloud','frame_index':i,
                        **dict(zip(('x_mm','y_mm','z_mm'),[round(float(v),4) for v in xyz])),
                        **dict(zip(('r','g','b'),rgb)),'pixel_u':uv[0],'pixel_v':uv[1]})
                if progress:progress(i,len(folders))
            summary=scan_asi_summary(summary_frames)
            writer.writerow({'record_type':'scan_summary','frame_count':len(summary_frames),
                **{'scan_'+k+'_mean_pct':summary['values'][k] for k in ('upper','lower','overall')},
                **{k+'_filter_samples':summary['used'][k] for k in ('upper','lower','overall')},
                'summary_method':'mean raw frame ASI; most frequent valid pair subset',
                **{k+'_pair_subset':str(summary.get('pairs',{}).get(k,'')) for k in ('upper','lower','overall')}})
            output.flush();os.fsync(output.fileno())
        os.replace(temp,final)
        return final,len(folders)
    except Exception:
        Path(temp).unlink(missing_ok=True);raise


def auto_export_capture(folder, destination=None):
    """Export after raw capture is saved; keep capture-time filter settings."""
    target=Path(destination) if destination is not None else AUTO_CSV_DIR
    target.mkdir(parents=True,exist_ok=True)
    return export_matlab_csv(folder,target)


def save_packet(packet,patient='Anonymous',root=DATA_ROOT,save_photo=False,store_depth=True):
    """CSV geometry, atomically published; optional explicit full-photo storage."""
    import shutil
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    pose_name=packet.pose_mode if packet.pose_mode in CAPTURE_MODES else 'Unspecified'
    import uuid
    scan=packet.device_meta.get('scan_id','legacy')
    name=f'{scan}_{safe_id(patient)}_{pose_name.upper()}_'+uuid.uuid4().hex[:10]+('_DEMO' if packet.demo else '')
    destination=root/name
    if destination.exists():raise FileExistsError('Capture already saved')
    temp=root/('.'+name+'.partial');temp.mkdir(exist_ok=False)
    try:
        data=np.column_stack([packet.points,packet.colors,packet.pixels])
        np.savetxt(temp/'cloud.csv',data,delimiter=',',header='x_mm,y_mm,z_mm,r,g,b,pixel_u,pixel_v',comments='',fmt=['%.6f','%.6f','%.6f','%d','%d','%d','%d','%d'])
        save_dense_landmarks(packet,temp/'dense_landmarks.csv')
        save_expression(packet,temp/'expression.csv')
        with (temp/'landmarks.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['landmark_id','x_mm','y_mm','z_mm','pixel_u','pixel_v','valid_depth_pixels','patch_depth_spread_mm'])
            for k in LM:
                p=packet.landmarks.get(k);u,v=packet.landmark_pixels.get(k,('',''));q=packet.quality.get(k,{})
                w.writerow([k,*(p if p is not None else ['','','']),u,v,q.get('valid_pixels',''),q.get('depth_spread_mm','')])
        with (temp/'depth_quality.csv').open('w',newline='') as f:
            fields=['landmark_id','status','source','valid_pixels','patch_radius','depth_spread_mm','median_depth_mm']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
            for k,q in packet.quality.items():w.writerow({'landmark_id':k,**q})
        with (temp/'metrics.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['metric','value']);w.writerows(scalar_metrics(packet).items())
        with (temp/'pair_measurements.csv').open('w',newline='') as f:
            fields=['left','right','span_mm','old_left_mm','old_right_mm','old_pair_pct','horizontal_mm','vertical_mm','depth_mm','hv_residual_mm','xyz_residual_mm','signed_horizontal_mm','signed_vertical_mm','signed_depth_mm']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(packet.scores['rows'])
        with (temp/'regional_asi_pairs.csv').open('w',newline='') as f:
            fields=['region','name','left','right','valid','reason','width_mm','dH_mm','dV_mm','dD_mm','pair_asi']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(packet.scores['asi']['pairs'])
        meta={'version':VERSION,'pose_mode':packet.pose_mode,'pose_verified':False,
              'upper_3d_asi':packet.scores['asi']['upper'],
              'lower_3d_asi':packet.scores['asi']['lower'],
              'overall_3d_asi':packet.scores['asi']['overall'],
              'landmark_ids_total':len(LM),'dense_detected_2d':len(packet.dense_pixels),'dense_valid_3d':len(packet.dense_landmarks),
              'dense_model':'MediaPipe FaceMesh indices; XYZ from measured stereo depth, not model Z',
              'landmarks_valid_3d':landmark_tally(packet.landmarks)['valid_3d'],
              'new_regional_asi_definition':'experimental bilateral H/V/D pooled widths; no midline M; partial-pair coverage tracked',
              'patient_id':safe_id(patient),'timestamp_s':packet.timestamp,'timestamp_iso':datetime.fromtimestamp(packet.timestamp).isoformat(),'wall_clock_verified':False,
              'host_sequence':packet.seq,'demo':int(packet.demo),'geometry':'one synchronized frame; camera coordinates; millimeters',
              'fx':packet.intrinsics[0],'fy':packet.intrinsics[1],'cx':packet.intrinsics[2],'cy':packet.intrinsics[3],
              'capture_frames':len(packet.frames) if packet.frames else 1,'selected_frame':packet.selected_frame,
              'landmark_side_convention':'patient anatomical left/right; old calculation retains historical assignment',
              'landmark_proxies':'superciliare:334/105; upper forehead:10; trichion requires manual marking',
              'target_distance_mm':TARGET_MM,'tolerance_mm':TOLERANCE_MM,'in_target_at_capture':int(packet.guide.get('ready',False)),
              'point_sampling_stride':CLOUD_STRIDE,'landmark_depth':'same-frame radii 2,3,5; >=3 retained valid pixels; robust local median; spread <=40mm; caution recorded',
              'photo_storage':'capture_rgb.png' if save_photo and packet.rgb is not None else 'not saved; full RGB retained only in current session memory',
              'old_asi':'supplied compute_metrics_3d function unchanged; clinical protocol unverified',
              'new_asi':'experimental mirrored-pair mean residual / medial-canthus span; HV default; XYZ optional',
              'reference_limit':'reference landmark error biases score; reference canthi are not scored',**packet.device_meta}
        with (temp/'metadata.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['key','value']);w.writerows(meta.items())
        if save_photo and packet.rgb is not None:
            import cv2
            if not cv2.imwrite(str(temp/'capture_rgb.png'),packet.rgb,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise OSError('Could not save capture photograph')
        if store_depth and packet.depth is not None:
            np.savetxt(temp/'raw_depth_mm.csv',packet.depth,fmt='%d',delimiter=',')
        summary=scan_asi_summary(packet.frames or [packet])
        with (temp/'scan_asi_summary.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['region','mean_asi_pct','included_frames','total_frames','pair_subset','method'])
            for region in ('upper','lower','overall'):
                writer.writerow([region,summary['values'][region],summary['used'][region],summary['total'],';'.join(summary['pairs'][region]),'mean raw frame ASI; most frequent valid pair subset'])
        if packet.frames:
            fields=['metric','valid_frames','total_frames','mean','sd','min','max']
            with (temp/'burst_summary.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(burst_statistics(packet.frames))
            with (temp/'burst_asi_frames.csv').open('w',newline='') as f:
                fields=['frame_index','timestamp_s','pose_mode','upper_3d_asi_pct','lower_3d_asi_pct',
                        'overall_3d_asi_pct','upper_pairs','lower_pairs','overall_pairs','landmarks_valid_3d',
                        'landmarks_total','landmarks_missing','upper_asi_filtered_pct','lower_asi_filtered_pct','overall_asi_filtered_pct','filter_window','filter_method','upper_filter_samples','lower_filter_samples','overall_filter_samples']
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for index,frame in enumerate(packet.frames,1):
                    filtered=asi_display_summary(packet.frames[:index],int(packet.device_meta.get('smooth_window',5)),True,packet.device_meta.get('smooth_method','mean'))
                    score=frame.scores['asi'];count=landmark_tally(frame.landmarks,frame.landmark_pixels,frame.quality)
                    writer.writerow({'frame_index':index,'timestamp_s':frame.timestamp,'pose_mode':frame.pose_mode,
                                     'upper_3d_asi_pct':score['upper'],'lower_3d_asi_pct':score['lower'],
                                     'overall_3d_asi_pct':score['overall'],
                                     'upper_pairs':score['coverage']['upper'][0],
                                     'lower_pairs':score['coverage']['lower'][0],
                                     'overall_pairs':score['coverage']['overall'][0],
                                     'landmarks_valid_3d':count['valid_3d'],'landmarks_total':count['total'],
                                     'landmarks_missing':';'.join(count['missing']),
                                     'filter_window':filtered['window_n'],'filter_method':packet.device_meta.get('smooth_method','mean'),
                                     **{k+'_asi_filtered_pct':filtered['display'][k] for k in ('upper','lower','overall')},
                                     **{k+'_filter_samples':filtered['n_used'][k] for k in ('upper','lower','overall')}})
            with (temp/'burst_manifest.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['frame_index','folder','timestamp_s','device_depth_sequence','pose_mode'])
                for index,frame in enumerate(packet.frames):
                    saved=save_packet(frame,f'F{index:03d}',temp/'frames',save_photo,False)
                    w.writerow([index,str(saved.relative_to(temp)),frame.timestamp,frame.device_meta.get('depth_sequence',''),frame.pose_mode])
        temp.rename(destination);packet.path=destination
        if packet.frames:
            for frame in packet.frames:frame.path=destination/frame.path.relative_to(temp)
        return destination
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise



def export_csv_capture(folder,destination,progress=None):
    """Compatibility entry point: always export one self-contained MATLAB CSV."""
    return export_matlab_csv(folder,destination,progress)


def export_destinations():
    """Existing mounted drives on typical Raspberry Pi OS desktops."""
    drives=[]
    for root in (Path('/media'),Path('/run/media'),Path('/mnt')):
        if not root.is_dir():continue
        try:candidates=list(root.iterdir())
        except OSError:continue
        for entry in candidates:
            if entry.is_dir() and os.path.ismount(entry):drives.append(entry)
            elif entry.is_dir() and root.name!='mnt':
                try:
                    drives.extend(p for p in entry.iterdir() if p.is_dir() and os.path.ismount(p))
                except OSError:pass
    return sorted(set(drives),key=str)

def load_packet(folder):
    folder=Path(folder)
    with (folder/'metadata.csv').open(newline='') as f:meta={r['key']:r['value'] for r in csv.DictReader(f)}
    with (folder/'cloud.csv').open(newline='') as f:cloud=list(csv.DictReader(f))
    if cloud:
        values=np.array([[float(r[k]) for k in ['x_mm','y_mm','z_mm','r','g','b','pixel_u','pixel_v']] for r in cloud])
        xyz=values[:,:3].astype(np.float32);colors=np.clip(values[:,3:6],0,255).astype(np.uint8);pixels=values[:,6:8].astype(np.int32)
    else:xyz=np.empty((0,3),np.float32);colors=np.empty((0,3),np.uint8);pixels=np.empty((0,2),np.int32)
    points={};lp={};quality={}
    with (folder/'landmarks.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            k=r['landmark_id']
            if r['x_mm'] and r['y_mm'] and r['z_mm']:
                p=np.array([float(r[x]) for x in ['x_mm','y_mm','z_mm']],np.float32)
                if finite_point(p):points[k]=p
            if r['pixel_u'] and r['pixel_v']:lp[k]=(int(float(r['pixel_u'])),int(float(r['pixel_v'])))
            quality[k]={'valid_pixels':int(float(r['valid_depth_pixels'])) if r['valid_depth_pixels'] else 0,
                        'depth_spread_mm':float(r['patch_depth_spread_mm']) if r['patch_depth_spread_mm'] else None}
    if (folder/'depth_quality.csv').is_file():
        with (folder/'depth_quality.csv').open() as f:
            for r in csv.DictReader(f):quality.setdefault(r['landmark_id'],{}).update({k:v for k,v in r.items() if k in ('status','source','patch_radius')})
    if float(meta.get('version','40').split('.')[0])<41:
        for left,right in OLD_PAIRS[1:]:
            for mapping in (points,lp,quality):
                a=mapping.pop(left,None);b=mapping.pop(right,None)
                if b is not None:mapping[left]=b
                if a is not None:mapping[right]=a
    intr=tuple(float(meta[k]) for k in ['fx','fy','cx','cy'])
    device_meta={k:float(meta[k]) for k in ['rgb_sequence','depth_sequence','sync_delta_ms','elapsed_s','smooth_window','monotonic_s'] if k in meta}
    device_meta.update({k:meta[k] for k in ['scan_id','subject_id','smooth_method','cloud_sampling',
                                             'camera_model','camera_platform','depthai_version','depth_unit',
                                             'ir_drivers','ir_dot_intensity','ir_flood_intensity',
                                             'rgb_depth_aligned','stereo_extended_disparity'] if k in meta})
    packet=Packet(int(meta['host_sequence']),float(meta['timestamp_s']),device_meta,intr,xyz,colors,pixels,points,lp,quality,path=folder,demo=meta.get('demo')=='1',pose_mode=meta.get('pose_mode','Unspecified'))
    load_dense_landmarks(packet,folder/'dense_landmarks.csv')
    if (folder/'capture_rgb.png').is_file():
        import cv2
        packet.rgb=cv2.imread(str(folder/'capture_rgb.png'))
    if (folder/'raw_depth_mm.csv').is_file():packet.depth=np.loadtxt(folder/'raw_depth_mm.csv',delimiter=',',dtype=np.uint16)
    if (folder/'manual_landmarks.csv').is_file():
        with (folder/'manual_landmarks.csv').open() as f:
            for r in csv.DictReader(f):
                k=r['landmark_id'];points[k]=np.array([float(r[n]) for n in ['x_mm','y_mm','z_mm']],np.float32)
                lp[k]=(int(r['pixel_u']),int(r['pixel_v']));quality[k]={'status':r['status'],'source':'manual photo mark'}
    if (folder/'burst_manifest.csv').is_file():
        with (folder/'burst_manifest.csv').open() as f:packet.frames=[folder/r['folder'] for r in csv.DictReader(f)]
        packet.selected_frame=int(meta.get('selected_frame',0))
    packet.scores=extended_metrics(points);packet.guide=make_guide(points,quality,lp,asi=packet.scores['asi']);load_expression(packet,folder/'expression.csv');return packet

# Shared geometry drives BOTH the RGB overlay and the interactive 3D review.
# Labels refer to actual 3D distances, never image-pixel distances.
def illustration_segments(packet,mode='new',selected_pair='CHEILION',include_depth=False,measure_pair=None):
    points=packet.landmarks;m=packet.scores;segments=[]
    if measure_pair:
        a,b=measure_pair
        if point_distance(points,a,b) is not None:
            segments.append((points[a],points[b],(40,211,255),f'{point_distance(points,a,b):.1f} mm',3))
    if mode in ('mouth','brow'):
        anchor='PRONASALE' if mode=='mouth' else ('TRICHION' if 'TRICHION' in points else 'FOREHEAD_PROXY')
        pair=('CHEILION_L','CHEILION_R') if mode=='mouth' else ('SUPERCILIARE_L','SUPERCILIARE_R')
        for name,side,color in zip(pair,('L','R'),((40,211,255),(255,185,65))):
            distance=point_distance(points,anchor,name)
            if distance is not None:segments.append((points[anchor],points[name],color,f'{side} {distance:.1f} mm',3))
    elif mode=='old' and m['old']:
        old=m['old'];n=old['plane_normal'];o=old['plane_point']
        for row in m['rows']:
            for key,label in [('left','L'),('right','R')]:
                name=row[key];p=points.get(name)
                if p is None:continue
                foot=p-np.dot(p-o,n)*n
                segments.append((p,foot,(222,202,35),f'{label} {np.linalg.norm(p-foot):.1f}',2))
    elif mode=='new':
        row=next((r for r in m['rows'][1:] if r['left'].startswith(selected_pair+'_')),None)
        if row and row['reflected'] is not None:
            basis=m['reference']['basis'];r=row['components'];start=row['reflected'];end=points[row['right']]
            h=start+r[0]*basis[:,0];v=h+r[1]*basis[:,1]
            segments.append((start,h,(245,208,34),f'H {abs(r[0]):.1f}',3))
            segments.append((h,v,(182,78,246),f'V {abs(r[1]):.1f}',3))
            segments.append((v,end,(77,190,247) if include_depth else (117,116,126),f'D {abs(r[2]):.1f}',2))
    return segments


def pair_statistics(frames,a,b):
    values=[point_distance(f.landmarks,a,b) for f in frames]
    good=np.array([v for v in values if v is not None and np.isfinite(v)],float)
    mean=float(good.mean()) if len(good) else None
    sd=float(good.std(ddof=1)) if len(good)>1 else None
    return {'values':values,'valid':len(good),'total':len(frames),'mean':mean,'sd':sd,
            'range':float(good.max()-good.min()) if len(good) else None,
            'cv':100*sd/mean if sd is not None and mean and mean>0 else None}

def fmt(value,unit='',digits=1):
    return 'Unavailable' if value is None or not np.isfinite(value) else f'{value:.{digits}f}{unit}'



def new_callout(img,project,packet,selected_pair,include_depth=False):
    import cv2
    row=next((r for r in packet.scores['rows'][1:] if r['left'].startswith(selected_pair+'_')),None)
    if row is None or row['components'] is None:return
    anchor=project(packet.landmarks[row['right']])
    if anchor is None:return
    h,w=img.shape[:2];ax,ay=anchor
    x=int(np.clip(ax+35,8,max(8,w-128)));y=int(np.clip(ay-24,35,max(35,h-82)))
    cv2.line(img,(int(ax),int(ay)),(x,y+18),(130,123,111),1,cv2.LINE_AA)
    cv2.rectangle(img,(x-5,y-20),(min(w-1,x+116),min(h-1,y+57)),(32,25,20),-1)
    for i,(name,key,color) in enumerate([('H','horizontal_mm',(245,208,34)),('V','vertical_mm',(182,78,246)),('D','depth_mm',(77,190,247))]):
        cv2.putText(img,f"{name}  {row[key]:.1f} mm",(x,y+i*22),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)


def photo_overlay(packet,mode='new',selected_pair='CHEILION',include_depth=False,measure_pair=None,show_landmarks=True):
    import cv2
    if packet.rgb is None:return None
    img=packet.rgb.copy();fx,fy,cx,cy=packet.intrinsics
    def project(p):
        if not finite_point(p) or p[2]<=0:return None
        return (int(np.clip(cx+p[0]*fx/p[2],-10000,10000)),int(np.clip(cy+p[1]*fy/p[2],-10000,10000)))
    for a,b,color,label,width in illustration_segments(packet,mode,selected_pair,include_depth,measure_pair):
        pa,pb=project(a),project(b)
        if pa is None or pb is None:continue
        cv2.line(img,pa,pb,color,width,cv2.LINE_AA)
        mid=((pa[0]+pb[0])//2,(pa[1]+pb[1])//2)
        if mode!='new':
            cv2.putText(img,label,(mid[0]+4,mid[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.38,(8,12,17),3,cv2.LINE_AA)
            cv2.putText(img,label,(mid[0]+4,mid[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    if mode=='new':new_callout(img,project,packet,selected_pair,include_depth)
    for k,p in (packet.landmarks.items() if show_landmarks else []):
        xy=project(p)
        if xy:cv2.circle(img,xy,3,(159,234,48),-1,cv2.LINE_AA)
    if packet.demo:cv2.putText(img,'SYNTHETIC DEMO - NOT PATIENT DATA',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(100,170,255),2)
    return img


class CloudRenderer:
    """Vectorized orthographic point viewer; no Matplotlib or Open3D loop."""
    def __init__(self):
        self.origin=None;self.fit_mm=110.;self.zoom=1.;self.yaw=0.;self.pitch=0.;self.pan=np.zeros(2);self.view_translation=np.zeros(3);self.roll=0.;self.perspective=False
        self.interacting=False;self.color_mode='solid';self.show_lines=False;self.show_labels=False;self.show_cloud=True;self.zrange=(200.,450.);self.hit_points={};self.show_landmarks=True;self.show_dense=False;self.mode='distance';self.selected_pair='CHEILION';self.include_depth=False
        self.measure_pair=('PRONASALE','CHEILION_R');self.last_size=(600,400)
    def reset(self,packet=None):
        self.origin=None;self.zoom=1.;self.yaw=0.;self.pitch=0.;self.pan[:]=0.;self.view_translation[:]=0.;self.roll=0.;self.perspective=False
        if packet is not None:self.fit(packet)
    def view_state(self):
        """Copy display-only state; never modify measured geometry or scores."""
        return {name:(value.copy() if isinstance(value,np.ndarray) else value)
                for name in ('origin','fit_mm','zoom','yaw','pitch','roll','pan',
                             'view_translation','perspective','zrange')
                for value in (getattr(self,name),)}
    def restore_view(self,state):
        for name,value in state.items():
            setattr(self,name,value.copy() if isinstance(value,np.ndarray) else value)
        self.interacting=False
    def fit(self,packet):
        points=packet.points
        if not len(points):points=np.array(list(packet.landmarks.values()))
        if not len(points):return
        points=points[np.isfinite(points).all(axis=1)]
        if not len(points):return
        self.origin=np.median(points,axis=0).astype(float)
        half=np.percentile(np.abs(points-self.origin),98,axis=0)
        self.fit_mm=float(max(70.,half[0],half[1])*1.15)
        lo,hi=np.percentile(points[:,2],[3,97]);self.zrange=(float(lo)-5,float(max(lo+1,hi))+5)
    def matrix(self):
        y,p=math.radians(self.yaw),math.radians(self.pitch);cy,sy=math.cos(y),math.sin(y);cp,sp=math.cos(p),math.sin(p)
        Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rx=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
        r=math.radians(self.roll);cr,sr=math.cos(r),math.sin(r);Rz=np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])
        return Rz@Rx@Ry
    def project(self,points,width,height):
        q=(np.asarray(points)-self.origin)@self.matrix().T+self.view_translation
        scale=min(width,height)*.43/self.fit_mm*self.zoom
        if self.perspective:
            distance=max(350.,self.fit_mm*4);scale=scale*distance/np.clip(distance+q[:,2],distance*.15,distance*8)
        xy=np.column_stack([width/2+q[:,0]*scale+self.pan[0]*width,height/2+q[:,1]*scale+self.pan[1]*height])
        return xy,q[:,2],scale
    def draw(self,packet,width=600,height=400,depth_color=True):
        import cv2
        width,height=max(120,int(width)),max(100,int(height));self.last_size=(width,height);self.hit_points={}
        img=np.full((height,width,3),(29,21,15),np.uint8)
        if packet is None:
            cv2.putText(img,'Waiting for measured XYZ...',(20,height//2),cv2.FONT_HERSHEY_SIMPLEX,.55,(174,166,150),1);return img
        if self.origin is None:self.fit(packet)
        if self.origin is None:return img
        for x in range(0,width,40):cv2.line(img,(x,0),(x,height),(40,31,24),1)
        for y in range(0,height,40):cv2.line(img,(0,y),(width,y),(40,31,24),1)
        points=packet.points
        if len(points) and self.show_cloud:
            # Stable pixel hash avoids row-striped thinning of organized depth.
            budget=min(MAX_RENDER_POINTS,12000) if self.interacting else MAX_RENDER_POINTS
            if len(points)>budget:
                uv=packet.pixels.astype(np.uint64);hashed=(uv[:,0]*73856093)^ (uv[:,1]*19349663)
                hashed=(hashed ^ (hashed>>13))*1274126177
                indices=np.argpartition(hashed,budget-1)[:budget]
            else:indices=np.arange(len(points))
            p=points[indices];rgb=packet.colors[indices];valid=np.isfinite(p).all(axis=1);p,rgb=p[valid],rgb[valid]
            xy,z,_=self.project(p,width,height);px=np.rint(xy[:,0]).astype(int);py=np.rint(xy[:,1]).astype(int)
            valid=(px>=1)&(px<width-1)&(py>=1)&(py<height-1);px,py,z=px[valid],py[valid],z[valid];rgb=rgb[valid]
            if self.color_mode=='solid':rgb=np.tile([77,202,205],(len(rgb),1))
            if self.color_mode=='depth' and depth_color:
                zz=p[valid,2];lo,hi=self.zrange;t=np.clip((zz-lo)/(hi-lo),0,1)
                palette=np.array([[249,213,78],[88,229,185],[63,170,243],[195,124,248]],float)
                u=t*3;ii=np.minimum(2,u.astype(int));frac=(u-ii)[:,None];rgb=palette[ii]*(1-frac)+palette[ii+1]*frac
            colors=np.clip(rgb[:,::-1],0,255).astype(np.uint8)
            # Four-pixel splats with deterministic nearest-depth selection.
            if len(px):
                px=np.concatenate([px,px+1,px,px+1]);py=np.concatenate([py,py,py+1,py+1]);zz=np.tile(z,4);cc=np.tile(colors,(4,1))
                order=np.argsort(zz);flat=py[order]*width+px[order];_,first=np.unique(flat,return_index=True);keep=order[first]
                img[py[keep],px[keep]]=cc[keep]
        for a,b,color,label,thick in (illustration_segments(packet,'distance',self.selected_pair,self.include_depth,self.measure_pair) if self.show_lines and self.mode=='distance' else []):
            xy,_,_=self.project(np.array([a,b]),width,height)
            pa,pb=[tuple(int(np.clip(v,-10000,10000)) for v in p) for p in xy]
            cv2.line(img,pa,pb,color,thick,cv2.LINE_AA)
            x,y=((xy[0]+xy[1])/2).astype(int)
            if self.mode in ('mouth','brow'):
                offset=18 if xy[1,0]>=xy[0,0] else -112
                x=int(np.clip(xy[1,0]+offset,4,max(4,width-112)));y=int(np.clip(xy[1,1]+25,18,height-5))
                cv2.line(img,pb,(x,y-6),(110,110,100),1,cv2.LINE_AA)
            if self.mode!='new' and 0<=x<width-90 and 12<y<height:
                cv2.putText(img,label,(x+4,y-5),cv2.FONT_HERSHEY_SIMPLEX,.4,(13,15,19),3,cv2.LINE_AA)
                cv2.putText(img,label,(x+4,y-5),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        if self.show_dense and packet.dense_landmarks:
            xy,_,_=self.project(np.array(list(packet.dense_landmarks.values())),width,height)
            for x,y in xy:
                if 0<=x<width and 0<=y<height:cv2.circle(img,(int(x),int(y)),2,(70,225,240),-1,cv2.LINE_AA)
        if self.show_landmarks or self.measure_pair:
            for name,p in packet.landmarks.items():
                if not finite_point(p) or (not self.show_landmarks and name not in self.measure_pair):continue
                xy,_,_=self.project(np.array([p]),width,height);x,y=xy[0]
                if not(0<=x<width and 0<=y<height):continue
                self.hit_points[name]=(float(x),float(y));selected=name in self.measure_pair or (self.mode=='mouth' and name in ('PRONASALE','CHEILION_L','CHEILION_R')) or (self.mode=='brow' and name in ('TRICHION','FOREHEAD_PROXY','SUPERCILIARE_L','SUPERCILIARE_R'))
                color=((55,165,255) if name in self.measure_pair[:1] else (240,110,190)) if selected else (180,180,180)
                cv2.circle(img,(int(x),int(y)),7 if selected else 4,(18,17,13),-1,cv2.LINE_AA)
                cv2.circle(img,(int(x),int(y)),5 if selected else 3,color,-1,cv2.LINE_AA)
                if self.show_labels:
                    text=({'PRONASALE':'Pn: nose tip','CHEILION_L':'Ch L','CHEILION_R':'Ch R','TRICHION':'Tr: hairline','SUPERCILIARE_L':'Sc L: brow','SUPERCILIARE_R':'Sc R: brow','FOREHEAD_PROXY':'Forehead proxy'}.get(name,PRETTY.get(name,name)));tx=int(np.clip(x+8,4,max(4,width-155)));ty=int(np.clip(y-8,15,height-8))
                    cv2.putText(img,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.38,(10,12,17),3,cv2.LINE_AA)
                    cv2.putText(img,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
        # Axis triad rotates with the data; cloud geometry stays camera-relative.
        axes=np.eye(3)@self.matrix().T;anchor=np.array([35.,height-45.]);colors=[(97,111,255),(129,224,74),(250,167,70)]
        for i,label in enumerate('XYZ'):
            tip=anchor+axes[i,:2]*28;cv2.line(img,tuple(anchor.astype(int)),tuple(tip.astype(int)),colors[i],2)
            cv2.putText(img,label,tuple((tip+3).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.36,colors[i],1)
        cv2.putText(img,f'XYZ / mm  {self.zoom:.1f}x  yaw {self.yaw:.0f} pitch {self.pitch:.0f}',(10,19),cv2.FONT_HERSHEY_SIMPLEX,.38,(179,165,141),1)
        if packet.demo:cv2.putText(img,'DEMO - SYNTHETIC',(max(10,width-170),height-10),cv2.FONT_HERSHEY_SIMPLEX,.43,(120,180,255),1)
        return img


def self_test():
    import tempfile, cv2
    p=demo_packet(0);r=face_reference(p.landmarks);origin,basis=r['origin'],r['basis'];points={k:v.copy() for k,v in p.landmarks.items()}
    # Exact symmetric pairs in the reference frame; projected midline stays unchanged.
    for left,right in OLD_PAIRS:
        a=(points[left]-origin)@basis;b=(points[right]-origin)@basis;span=abs(b[0]-a[0])/2;yz=(a[1:]+b[1:])/2
        points[left]=origin+np.array([-span,*yz])@basis.T;points[right]=origin+np.array([span,*yz])@basis.T
    base=metrics(points);assert base['new']['hv_pct']<1e-5
    changed={k:v.copy() for k,v in points.items()};changed['CHEILION_R']+=10*basis[:,1]
    m=metrics(changed);assert m['rows'][2]['signed_vertical_mm']>9.99;expected=100*(10/3)/r['scale_mm'];assert np.isclose(m['new']['hv_pct'],expected,atol=1e-5)
    depth={k:v.copy() for k,v in points.items()};depth['CHEILION_R']+=10*basis[:,2]
    d=metrics(depth);assert d['new']['hv_pct']<1e-5 and np.isclose(d['new']['xyz_pct'],expected,atol=1e-5)
    angle=.52;R=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
    moved={k:R@v+np.array([20.,30.,80.]) for k,v in changed.items()};assert np.isclose(metrics(moved)['new']['hv_pct'],m['new']['hv_pct'],atol=1e-5)
    missing=dict(points);del missing['CHEILION_R'];assert metrics(missing)['new']['hv_pct'] is None
    assert point_distance(points,'PRONASALE','PRONASALE') == 0
    renderer=CloudRenderer();times=[]
    for mode in ['distance','old','new']:
        renderer.mode=mode;t=time.perf_counter();image=renderer.draw(p,600,400);times.append(1000*(time.perf_counter()-t));assert image.shape==(400,600,3)
    before=renderer.origin.copy();q=demo_packet(20);q.points+=np.array([10,0,0]);renderer.draw(q);assert np.array_equal(renderer.origin,before)
    with tempfile.TemporaryDirectory() as root:
        saved=save_packet(p,'Unit Test',root);loaded=load_packet(saved)
        assert np.allclose(loaded.points,p.points,atol=1e-5)
        assert np.isclose(loaded.scores['new']['hv_pct'],p.scores['new']['hv_pct'],atol=1e-5)
        assert loaded.rgb is None and all(f.suffix=='.csv' for f in saved.iterdir())
    with tempfile.TemporaryDirectory() as root:
        saved=save_packet(p,'Photo test',root,save_photo=True);loaded=load_packet(saved)
        assert np.array_equal(loaded.rgb,p.rgb) and (saved/'capture_rgb.png').is_file()
    # v48 new regional ASI regression tests: symmetry, H/V/D sensitivity,
    # rigid pose invariance, missing values, 3 expression metadata modes.
    p=demo_packet(0)
    origin=asi_reference(p.landmarks)['origin'];basis=asi_reference(p.landmarks)['basis']
    symmetrical={k:np.array(v,dtype=float).copy() for k,v in p.landmarks.items()}
    for region,pairs in ASI_GROUPS.items():
        for left,right,_ in pairs:
            x=(symmetrical[left]-origin)@basis;y=(symmetrical[right]-origin)@basis
            half=(abs(x[0])+abs(y[0]))/2
            average=(x[1:]+y[1:])/2
            symmetrical[left]=origin+np.array([-half,*average])@basis.T
            symmetrical[right]=origin+np.array([half,*average])@basis.T
    base=compute_three_region_asi(symmetrical)
    assert max(base[k] for k in ('upper','lower','overall'))<1e-5,base
    for coord in range(3):
        moved={k:v.copy() for k,v in symmetrical.items()}
        moved['CHEILION_R']=moved['CHEILION_R']+8*basis[:,coord]
        changed=compute_three_region_asi(moved)
        assert changed['lower']>0 and changed['overall']>0,(coord,changed)
        assert changed['upper']<1e-5
        if coord==2:assert changed['components']['lower']['D']>0
        if coord==1:assert changed['components']['lower']['V']>0
    angle=.41;rotation=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
    posed={k:rotation@v+np.array([15.,21.,36.]) for k,v in moved.items()}
    assert np.isclose(compute_three_region_asi(posed)['overall'],changed['overall'],atol=1e-5)
    no_brow=dict(symmetrical);no_brow.pop('SUPERCILIARE_L')
    partial=compute_three_region_asi(no_brow)
    assert partial['upper'] is not None and partial['coverage']['upper']==(2,3)
    missing_ref=dict(symmetrical);missing_ref.pop('SUBNASALE')
    assert compute_three_region_asi(missing_ref)['overall'] is None
    count=landmark_tally(p.landmarks,p.landmark_pixels,p.quality)
    assert count['total']==len(LM)==21 and count['valid_3d']==18 and count['detected_2d']==18
    from dataclasses import replace
    with tempfile.TemporaryDirectory() as root:
        for pose in CAPTURE_MODES:
            example=replace(p,pose_mode=pose)
            saved=save_packet(example,'Mode test',root,save_photo=False)
            with (saved/'metadata.csv').open() as f:metadata={row['key']:row['value'] for row in csv.DictReader(f)}
            assert metadata['pose_mode']==pose
            assert (saved/'regional_asi_pairs.csv').exists()
            restored=load_packet(saved)
            assert restored.pose_mode==pose
            for key in ('upper','lower','overall'):
                assert np.isclose(example.scores['asi'][key],restored.scores['asi'][key],atol=1e-4)
        examples=[replace(demo_packet(i),pose_mode='Blinking') for i in range(3)]
        group=replace(examples[0],frames=examples,selected_frame=0)
        directory=save_packet(group,'Burst test',root,save_photo=False,store_depth=False)
        assert (directory/'burst_asi_frames.csv').is_file()
        with (directory/'burst_asi_frames.csv').open() as f:records=list(csv.DictReader(f))
        assert len(records)==3 and all(item['pose_mode']=='Blinking' for item in records)
        assert all(item['upper_3d_asi_pct'] for item in records)
    print('PASS v48: 3D ASI symmetry/H/V/D, pose invariance, partial coverage, missing reference, counts, all 3 expression labels and saved 3-frame ASIs.')
    print('PASS: symmetry, vertical and depth sensitivity, pose invariance, missing landmarks, fixed view, CSV round-trip, and 3 render modes.')
    print('Renderer ms (this machine, NOT Raspberry Pi):',', '.join(f'{t:.1f}' for t in times))


def v53_regression_test():
    import sys
    m=sys.modules[__name__]
    import numpy as np, tempfile,csv,copy
    from pathlib import Path
    from dataclasses import replace
    from types import SimpleNamespace
    p=m.demo_packet()
    frames=[]
    for i,value in enumerate([2,4,30,6,8]):
     score=copy.deepcopy(p.scores)
     for region in ('upper','lower','overall'):score['asi'][region]=value
     frames.append(replace(p,seq=i,timestamp=float(i),scores=score))
    a=m.asi_display_summary(frames,3)
    assert a['display']['overall']==44/3
    assert m.asi_display_summary(frames,3,True,'median')['display']['overall']==8
    assert m.asi_display_summary(frames,1)['display']['overall']==8
    assert m.asi_display_summary(frames,3,False)['display']['overall']==8
    assert m.asi_display_summary(frames[:2],90)['n_used']['overall']==2
    # Causal frame boundary: an arbitrarily changed future frame never changes frame 2.
    assert m.asi_display_summary(frames[:2],3)['display']['overall']==3
    frames[-1].scores['asi']['overall']=None
    assert m.asi_display_summary(frames,3)['display']['overall'] is None
    frames[-1].scores['asi']['overall']=8
    # Same count with a different valid pair subset must not be combined.
    valid=next(r for r in frames[-1].scores['asi']['pairs'] if r['valid'])
    valid['name']='different-pair'
    assert m.asi_display_summary(frames,5)['n_used']['overall']==1
    # 468 dense points: all detected pixels stored, XYZ only with measured depth.
    face=[SimpleNamespace(x=.25+(i%26)*.018,y=.22+(i//26)*.025) for i in range(468)]
    mesh=SimpleNamespace(process=lambda _:SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=face)]))
    rgb=np.zeros((480,640,3),np.uint8);depth=np.full((480,640),350,np.uint16)
    q=m.analyze_frame(rgb,depth,(500,500,320,240),mesh,0,{})
    assert len(q.dense_pixels)==468 and len(q.dense_landmarks)==468
    q2=m.analyze_frame(rgb,np.zeros_like(depth),(500,500,320,240),mesh,1,{})
    assert len(q2.dense_pixels)==468 and not q2.dense_landmarks
    # Store/export/reload 90 frames with dense points and both ASI columns.
    with tempfile.TemporaryDirectory() as tmp:
     root=Path(tmp);out=root/'export';out.mkdir()
     series=[]
     for i in range(90):
      f=m.compact_capture_frame(m.demo_packet(i),'Smiling',100)
      f.dense_pixels={0:(100,120),1:(130,140)};f.dense_landmarks={0:np.array([1.,2.,350.])}
      f.rgb=None;f.depth=None
      f.device_meta.update(scan_id='123',smooth_window=20,smooth_method='mean',elapsed_s=i*.1)
      series.append(f)
     packet=replace(series[-1],frames=series,selected_frame=89)
     folder=m.save_packet(packet,'Demo',root/'capture',store_depth=False)
     loaded=m.load_packet(folder)
     assert len(loaded.frames)==90 and len(loaded.dense_pixels)==2 and len(loaded.dense_landmarks)==1
     path,count=m.export_matlab_csv(folder,out,filter_window=20)
     assert count==90
     with path.open() as f:rows=list(csv.DictReader(f))
     landmarks=[r for r in rows if r['record_type']=='landmark']
     dense=[r for r in rows if r['record_type']=='dense_landmark']
     assert len(landmarks)==90*21 and len(dense)==180
     last=landmarks[-1]
     expected=m.asi_display_summary(series,20)['display']['overall']
     assert np.isclose(float(last['overall_asi_filtered_pct']),expected,atol=1e-4)
     assert int(last['overall_filter_samples'])==20
     assert dense[1]['x_mm']=='' and dense[1]['pixel_u']=='130'
     assert not list(folder.rglob('*.png'))
     # Compatibility: legacy captures have no dense file.
     (folder/'dense_landmarks.csv').unlink()
     assert not m.load_packet(folder).dense_pixels
    print('PASS v53: causal mean/median, window changes, warm-up, invalid and changed coverage, 468-point measured-depth extraction, 90-frame save/reload/export, CSV-only defaults, legacy files.')


def v54_regression_test():
    import copy
    from dataclasses import replace
    frame=demo_packet();frames=[]
    for value in (2.,4.,6.,100.):
        score=copy.deepcopy(frame.scores)
        for region in ('upper','lower','overall'):score['asi'][region]=value
        frames.append(replace(frame,scores=score))
    # A missing reference excludes that frame, not a fabricated zero.
    for region in ('upper','lower','overall'):frames[-1].scores['asi'][region]=None
    result=scan_asi_summary(frames)
    assert result['values']['overall']==4. and result['used']['overall']==3
    assert result['total']==4
    assert scan_asi_summary([])['values']['overall'] is None
    # Different pair coverage cannot bias a single recording result.
    frames[-1].scores['asi']['overall']=100.
    next(r for r in frames[-1].scores['asi']['pairs'] if r['valid'])['name']='different pair'
    assert scan_asi_summary(frames)['values']['overall']==4.
    assert compute_metrics_3d({}) is None
    missing=dict(frame.landmarks);missing.pop('MENTON',None)
    assert extended_metrics(missing)['new']['core_hv_pct'] is None
    print('PASS v54: whole-recording means, invalid-frame exclusion, fixed pair coverage and legacy missing-depth behavior.')


def camera_check(ir_dot=0.35,ir_flood=0.10,seconds=10.0):
    """Hardware smoke test: no UI/MediaPipe, just aligned frames and IR control."""
    source=OakSource(ir_dot=ir_dot,ir_flood=ir_flood)
    try:
        print(f'Device: {source.name}; backend: {source.kind}; platform: {source.platform}')
        print(f'Calibrated RGB intrinsics (fx,fy,cx,cy): {source.intrinsics}')
        print('Looking for synchronized 640x480 RGB + uint16 depth frames...')
        samples=[];t0=time.monotonic()
        while time.monotonic()-t0<seconds and len(samples)<10:
            item=source.read()
            if item is None:time.sleep(.04);continue
            rgb,depth,meta=item
            # Nearby facial depth only; failure of this coverage test on an empty
            # desk/background does NOT mean the camera is broken.
            center=depth[120:360,160:480]
            valid=center[(center>=180)&(center<=1500)]
            coverage=100.0*len(valid)/center.size
            median=float(np.median(valid)) if len(valid) else float('nan')
            samples.append((meta['sync_delta_ms'],coverage,median))
        if not samples:
            raise RuntimeError('No synchronized RGB/depth frames received in 10 s. '
                               'Check the USB 3 cable, permissions, power, and DepthAI version.')
        print(f'PASS: {len(samples)} RGB/depth pairs, '
              f'max RGB/depth timestamp difference {max(x[0] for x in samples):.1f} ms.')
        print(f'Center-region valid 0.18–1.50m depth: '
              f'{np.mean([x[1] for x in samples]):.1f}% mean coverage.')
        if all(x[1]<1. for x in samples):
            print('WARNING: Almost no valid depth near the center. Test with a textured '
                  'target or face at 30–40 cm; inspect IR emitter and exposure settings.')
        else:
            depths=[x[2] for x in samples if math.isfinite(x[2])]
            print(f'Median valid center depth: {np.median(depths):.0f} mm.')
        print('NOTE: Device connection and stream delivery were checked; millimeter '
              'accuracy and facial asymmetry validity have NOT been validated.')
    finally:source.close()


def camera_configuration_self_test():
    """Test host-side IR calls and synchronized-frame guards without hardware."""
    from types import SimpleNamespace
    from datetime import timedelta as _td
    calls=[]
    class DummyDevice:
        def getProductName(self):return 'OAK-D Pro (mock)'
        def getPlatform(self):return 'RVC2'
        def getIrDrivers(self):return [('LM3644',2,99)]
        def setIrLaserDotProjectorIntensity(self,n):calls.append(('dot',n));return True
        def setIrFloodLightIntensity(self,n):calls.append(('flood',n));return True
    fake=object.__new__(OakSource)
    fake.device=DummyDevice();fake.ir_dot_requested=.35;fake.ir_flood_requested=.1
    fake.ir_dot=0.;fake.ir_flood=0.;fake.ir_drivers=[]
    fake.depthai_version='3.fake';fake.kind='test';fake.name='OAK camera';fake.platform='unknown';fake.extended_disparity=True
    fake._set_device_details(fake.device);fake._enable_ir()
    assert fake.name.startswith('OAK-D Pro') and fake.ir_drivers==['LM3644']
    assert calls==[('dot',.35),('flood',.1)]
    frame_rgb=np.zeros((480,640,3),np.uint8)
    frame_depth=np.full((480,640),350,np.uint16)
    class DummyFrame:
        def __init__(self,frame,seq=10):self.frame=frame;self.seq=seq
        def getCvFrame(self):return self.frame
        def getFrame(self):return self.frame
        def getSequenceNum(self):return self.seq
        def getTimestampDevice(self):return _td(seconds=self.seq/20.)
    class DummyQueue:
        def tryGet(self):return {'rgb':DummyFrame(frame_rgb),'depth':DummyFrame(frame_depth)}
    fake.queue=DummyQueue()
    rgb,depth,meta=fake.read()
    assert rgb.shape==(480,640,3) and depth.shape==(480,640)
    assert meta['ir_dot_intensity']==.35 and meta['ir_flood_intensity']==.1
    assert meta['rgb_depth_aligned']==1 and meta['sync_delta_ms']==0
    from dataclasses import replace
    import tempfile
    packet=demo_packet()
    packet.device_meta.update(meta)
    with tempfile.TemporaryDirectory() as temp:
        saved=save_packet(packet,'OAK Pro mock',temp,store_depth=False)
        reloaded=load_packet(saved)
        assert reloaded.device_meta['camera_model']=='OAK-D Pro (mock)'
        assert float(reloaded.device_meta['ir_dot_intensity'])==.35
        assert float(reloaded.device_meta['ir_flood_intensity'])==.10
    class FailedDevice(DummyDevice):
        def setIrLaserDotProjectorIntensity(self,n):return False
    fake.device=FailedDevice()
    try:fake._enable_ir()
    except RuntimeError as exc:assert 'refused' in str(exc)
    else:raise AssertionError('Must fail if IR was requested but unavailable')
    print('PASS v59: camera metadata, successful IR activation, unsupported-IR error, '
          'and synchronized RGB/depth frame validation (mock, no camera).')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windowed',action='store_true');parser.add_argument('--demo',action='store_true');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--camera-check',action='store_true',
        help='Without starting Kivy/MediaPipe, verify camera, IR emitters, synced RGB/depth, and depth coverage')
    parser.add_argument('--ir-dot',type=float,default=0.35,metavar='0..1',
        help='OAK-D Pro active stereo laser dot intensity, normalized 0..1 (default 0.35; 0 disables)')
    parser.add_argument('--ir-flood',type=float,default=0.10,metavar='0..1',
        help='OAK-D Pro mono IR floodlight intensity, normalized 0..1 (default 0.10; 0 disables)')
    parser.add_argument('--target-cm',type=float,default=35.);parser.add_argument('--tolerance-cm',type=float,default=5.)
    parser.add_argument('--cloud-stride',type=int,choices=(1,2,3),default=1,help='Depth pixel sampling stride; 1 is full pixel density')
    parser.add_argument('--render-points',type=int,default=32000,help='Maximum displayed cloud points (1000..200000); measurement XYZ unchanged')
    parser.add_argument('--frames',type=int,default=90,help='Independent frames in one expression (1..300); default 90')
    parser.add_argument('--pose',choices=CAPTURE_MODES,default='Neutral',help='Initial expression mode; also selectable on touch screen')
    parser.add_argument('--save-photos',action='store_true',default=False,help='Explicitly enable stored RGB photographs')
    parser.add_argument('--no-photos',dest='save_photos',action='store_false',help='Keep full photos only in session memory; geometry still saved')
    parser.add_argument('--export-capture',type=Path,help='Export one MATLAB CSV without opening the camera')
    parser.add_argument('--export-to',type=Path,help='Destination directory for --export-capture')
    parser.add_argument('--scans-dir',type=Path,help='Exact local Dropbox scans directory')
    parser.add_argument('--capture-cloud-points',type=int,default=8000,help='Stored cloud points/frame; measured landmarks unchanged')
    parser.add_argument('--screenshot-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    global TARGET_MM,TOLERANCE_MM,MAX_RENDER_POINTS,CLOUD_STRIDE
    TARGET_MM=args.target_cm*10;TOLERANCE_MM=args.tolerance_cm*10
    if TARGET_MM<=0 or TOLERANCE_MM<=0:parser.error('Distance target and tolerance must be positive')
    if any(not math.isfinite(x) or not (0.0<=x<=1.0) for x in (args.ir_dot,args.ir_flood)):
        parser.error('--ir-dot and --ir-flood must each be between 0.0 and 1.0')
    if not 1<=args.frames<=300:parser.error('--frames must be 1..300')
    if not 1000<=args.render_points<=200000:parser.error('--render-points must be 1000..200000')
    MAX_RENDER_POINTS=args.render_points;CLOUD_STRIDE=args.cloud_stride
    if args.export_capture:
        if not args.export_to:parser.error('--export-capture requires --export-to')
        path,count=export_matlab_csv(args.export_capture,args.export_to);print(f'Exported {count} frames in one CSV: {path}');return
    if args.export_to:parser.error('--export-to requires --export-capture')
    if args.capture_cloud_points<100:parser.error('--capture-cloud-points must be >=100')
    if args.scans_dir:save_settings(scans_dir=str(args.scans_dir.expanduser()))
    if args.self_test:
        self_test();v53_regression_test();v54_regression_test();camera_configuration_self_test();return
    if args.camera_check:
        camera_check(args.ir_dot,args.ir_flood);return
    run_ui(args)


def run_ui(args):
    # Configure before importing the window. Keep the existing Pi touch providers.
    from kivy.config import Config
    Config.set('graphics','fullscreen','0' if args.windowed else 'auto')
    Config.set('graphics','width','800');Config.set('graphics','height','480')
    Config.set('graphics','show_cursor','1' if args.windowed else '0')
    Config.set('input','mouse','mouse,multitouch_on_demand')
    Config.set('kivy','keyboard_mode','system' if args.windowed else 'systemanddock')
    Config.set('kivy','log_level','warning')
    from kivy.app import App
    from kivy.clock import Clock
    from kivy.core.window import Window
    from kivy.graphics import Color,RoundedRectangle,Rectangle,Line,Ellipse
    from kivy.uix.widget import Widget
    from kivy.uix.slider import Slider
    from kivy.graphics.texture import Texture
    from kivy.metrics import dp,sp
    from kivy.properties import ListProperty,NumericProperty
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.image import Image
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.screenmanager import Screen,ScreenManager,NoTransition
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.textinput import TextInput
    from kivy.uix.popup import Popup
    from kivy.uix.checkbox import CheckBox
    from kivy.uix.spinner import Spinner
    import cv2
    BG=(.035,.055,.085,1);CARD=(.075,.105,.15,1);TEAL=(.12,.77,.67,1)
    TEXT=(.93,.96,.98,1);MUTED=(.58,.67,.76,1);WARN=(1.,.71,.28,1);PINK=(.91,.36,.71,1)
    Window.clearcolor=BG

    def label(text='',size=15,color=TEXT,bold=False,**kw):
        kw.setdefault('halign','left');kw.setdefault('valign','middle')
        widget=Label(text=text,font_size=sp(size),color=color,bold=bold,**kw)
        widget.bind(size=lambda obj,value:setattr(obj,'text_size',value));return widget
    def backdrop(widget,color=CARD,radius=16):
        with widget.canvas.before:
            Color(*color);rect=RoundedRectangle(pos=widget.pos,size=widget.size,radius=[dp(radius)])
        widget.bind(pos=lambda obj,_:setattr(rect,'pos',obj.pos),size=lambda obj,_:setattr(rect,'size',obj.size))
    class TouchButton(Button):
        tint=ListProperty(CARD)
        def __init__(self,**kwargs):
            kwargs.setdefault('font_size',sp(15));kwargs.setdefault('color',TEXT)
            super().__init__(background_normal='',background_down='',background_color=(0,0,0,0),**kwargs)
            with self.canvas.before:
                self.shade=Color(*self.tint);self.shape=RoundedRectangle(pos=self.pos,size=self.size,radius=[dp(12)])
            self.bind(pos=self.paint,size=self.paint,tint=self.paint,state=self.paint,disabled=self.paint)
        def paint(self,*_):
            self.shape.pos=self.pos;self.shape.size=self.size
            self.shade.rgba=tuple(v*.8 if i<3 and self.state=='down' else v for i,v in enumerate(self.tint))
            self.opacity=.45 if self.disabled else 1.
    last_actions={}
    def button(text,callback,tint=CARD,height=44,**kw):
        kw.setdefault('size_hint_y',None)
        b=TouchButton(text=text,tint=tint,height=dp(height),always_release=True,**kw)
        action_key=(id(callback.__self__),callback.__name__) if getattr(callback,'__self__',None) is not None else text
        def fire(*_):
            now=time.monotonic()
            if now-last_actions.get(action_key,-1)<.12:return
            last_actions[action_key]=now;callback()
        b.bind(on_release=fire);return b
    def arrow_button(direction,callback,**kwargs):
        # Draw the icon: no reliance on font coverage for Unicode arrows.
        b=button('',callback,**kwargs)
        with b.canvas.after:
            Color(*TEXT);stem=Line(points=[0,0,0,0],width=2);head=Line(points=[0,0,0,0,0,0],width=2)
        def position(*_):
            d=np.array({'up':(0,1),'down':(0,-1),'left':(-1,0),'right':(1,0)}[direction],float)
            across=np.array([-d[1],d[0]]);center=np.array([b.x+b.width/2,b.y+b.height/2]);tip=center+d*dp(11);tail=center-d*dp(11)
            first=tip-d*dp(8)+across*dp(7);last=tip-d*dp(8)-across*dp(7)
            stem.points=[*tail,*tip];head.points=[*first,*tip,*last]
        b.bind(pos=position,size=position);position();return b
    class ArrowScroll(BoxLayout):
        """Vertical scrolling: tap/hold arrows or drag the thumb; swiping is optional."""
        scroll_y=NumericProperty(1.)
        def __init__(self,**kwargs):
            kwargs.pop('do_scroll_x',None);super().__init__(orientation='horizontal',spacing=dp(5),**kwargs)
            self.content_widget=None;self.repeat_event=None;self.delay_event=None;self.syncing=False
            self.scroller=ScrollView(do_scroll_x=False,bar_width=dp(8),scroll_type=['bars','content'],scroll_distance=dp(8),scroll_timeout=180)
            self.rail=BoxLayout(orientation='vertical',size_hint_x=None,width=dp(46),spacing=dp(5))
            self.up=arrow_button('up',lambda:None,height=46,tint=TEAL)
            self.down=arrow_button('down',lambda:None,height=46,tint=TEAL)
            self.thumb=Slider(orientation='vertical',min=0,max=1,value=1,step=0,size_hint_x=1)
            self.rail.add_widget(self.up);self.rail.add_widget(self.thumb);self.rail.add_widget(self.down)
            super().add_widget(self.scroller);super().add_widget(self.rail)
            self.up.bind(on_press=lambda *_:self.start_scroll(1),on_release=lambda *_:self.stop_scroll())
            self.down.bind(on_press=lambda *_:self.start_scroll(-1),on_release=lambda *_:self.stop_scroll())
            self.thumb.bind(value=self.thumb_changed);self.scroller.bind(scroll_y=self.content_scrolled)
            self.bind(scroll_y=self.position_changed,parent=lambda *_:self.stop_scroll() if self.parent is None else None)
            self.sync_trigger=Clock.create_trigger(self.update_range,0);self.scroller.bind(size=lambda *_:self.sync_trigger())
        def add_widget(self,widget,*args,**kwargs):
            self.content_widget=widget;self.scroller.add_widget(widget,*args,**kwargs)
            widget.bind(height=lambda *_:self.sync_trigger());self.sync_trigger()
        def content_scrolled(self,_,value):
            self.scroll_y=float(np.clip(value,0,1))
        def position_changed(self,_,value):
            if self.syncing:return
            self.syncing=True;self.scroller.scroll_y=float(np.clip(value,0,1));self.thumb.value=self.scroller.scroll_y;self.syncing=False;self.sync_trigger()
        def thumb_changed(self,_,value):
            if not self.syncing:self.scroll_y=value
        def update_range(self,*_):
            overflow=max(0,(self.content_widget.height if self.content_widget else 0)-self.scroller.height)
            self.rail.width=dp(46) if overflow>1 else 0;self.rail.opacity=1 if overflow>1 else 0;self.rail.disabled=overflow<=1
            self.up.disabled=overflow<=1 or self.scroll_y>=.99999
            self.down.disabled=overflow<=1 or self.scroll_y<=.00001
            if overflow<=1:self.scroll_y=1;self.stop_scroll()
        def step_scroll(self,direction):
            overflow=max(0,(self.content_widget.height if self.content_widget else 0)-self.scroller.height)
            if overflow>1:self.scroll_y=float(np.clip(self.scroll_y+direction*max(dp(100),self.scroller.height*.65)/overflow,0,1))
        def start_scroll(self,direction):
            self.stop_scroll();self.step_scroll(direction)
            ancestor=self.parent;visited=set()
            while ancestor is not None and id(ancestor) not in visited:
                visited.add(id(ancestor))
                if isinstance(ancestor,Popup):ancestor.bind(on_dismiss=lambda *_:self.stop_scroll());break
                ancestor=ancestor.parent
            def repeat(*_):
                self.delay_event=None;self.repeat_event=Clock.schedule_interval(lambda *_:self.step_scroll(direction),.11)
            self.delay_event=Clock.schedule_once(repeat,.3)
        def stop_scroll(self):
            for name in ('delay_event','repeat_event'):
                event=getattr(self,name,None)
                if event:event.cancel();setattr(self,name,None)
    def space(height=8):return BoxLayout(size_hint_y=None,height=dp(height))
    def notice(title,text):
        content=BoxLayout(orientation='vertical',padding=dp(14),spacing=dp(8))
        scroll=ArrowScroll(do_scroll_x=False);message=Label(text=text,font_size=sp(14),color=TEXT,halign='left',valign='top',size_hint_y=None)
        message.bind(width=lambda obj,width:setattr(obj,'text_size',(width,None)))
        message.bind(texture_size=lambda obj,size:setattr(obj,'height',size[1]+dp(12)))
        scroll.add_widget(message);content.add_widget(scroll)
        popup=Popup(title=title,content=content,size_hint=(.9,.88),auto_dismiss=False)
        content.add_widget(button('Close',popup.dismiss,tint=TEAL));popup.open();return popup
    class TracePlot(BoxLayout):
        def __init__(self,values,title,secondary=None,**kw):
            super().__init__(orientation='vertical',**kw)
            finite=[v for v in list(values)+list(secondary or []) if v is not None and np.isfinite(v)]
            lo=min(finite) if finite else 0;hi=max(finite) if finite else 1
            pad=max(1.,(hi-lo)*.15);lo-=pad;hi+=pad
            self.add_widget(label(f'{title}   {lo:.1f}–{hi:.1f}',13,height=dp(27),size_hint_y=None))
            plot=Widget();self.add_widget(plot)
            self.add_widget(label(f'Frame 1                                                   Frame {len(values)}',11,height=dp(20),size_hint_y=None,color=MUTED))
            def draw(*_):
                plot.canvas.clear()
                with plot.canvas:
                    Color(.07,.10,.15,1);Rectangle(pos=plot.pos,size=plot.size)
                    Color(.2,.27,.32,1)
                    for fraction in (.25,.5,.75):Line(points=[plot.x,plot.y+plot.height*fraction,plot.right,plot.y+plot.height*fraction],width=1)
                    for series,color in ((values,(1.,.65,.22,1)),(secondary or [],(.18,.85,.72,1))):
                        Color(*color);previous=None
                        for index,value in enumerate(series):
                            if value is None or not np.isfinite(value):previous=None;continue
                            x=plot.x+6+(plot.width-12)*index/max(1,len(values)-1);y=plot.y+6+(plot.height-12)*(value-lo)/(hi-lo)
                            if previous:Line(points=[*previous,x,y],width=1.4)
                            Ellipse(pos=(x-2,y-2),size=(4,4));previous=(x,y)
            plot.bind(pos=draw,size=draw)

    class FrameImage(Image):
        def __init__(self,**kw):
            super().__init__(allow_stretch=True,keep_ratio=True,**kw)
        def show(self,bgr):
            if bgr is None:return
            h,w=bgr.shape[:2]
            if self.texture is None or self.texture.size!=(w,h):
                self.texture=Texture.create(size=(w,h),colorfmt='rgb');self.texture.flip_vertical()
            self.texture.blit_buffer(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes(),colorfmt='rgb',bufferfmt='ubyte')
            self.canvas.ask_update()  # blit_buffer alone does not schedule a redraw
            self.redraw_count=getattr(self,'redraw_count',0)+1
    class PhotoView(FrameImage):
        def __init__(self,**kw):
            super().__init__(**kw);self.on_mark=None;self.missing_photo=False
        def show_review(self,packet,mode,pair,depth,measurement,landmarks,frame_text,dense=False):
            self.missing_photo=packet.rgb is None
            if self.missing_photo:
                image=np.full((480,640,3),(29,21,14),np.uint8)
                for text,y,size in [('No photo saved for '+frame_text,170,.7),('This frame still has 3D measurements.',220,.58),('Use a new capture for photos of every frame.',260,.55)]:
                    cv2.putText(image,text,(24,y),cv2.FONT_HERSHEY_SIMPLEX,size,(220,225,231),1,cv2.LINE_AA)
            else:image=photo_overlay(packet,mode,pair,depth,measurement,landmarks)
            if dense and not self.missing_photo:image=dense_overlay(image,packet)
            self.show(image)
        def on_touch_down(self,touch):
            if self.on_mark and self.collide_point(*touch.pos) and self.texture:
                w,h=self.texture.size;scale=min(self.width/w,self.height/h)
                ox=self.x+(self.width-w*scale)/2;oy=self.y+(self.height-h*scale)/2
                u=int((touch.x-ox)/scale);v=int(h-(touch.y-oy)/scale)
                if 0<=u<w and 0<=v<h:self.on_mark(u,v);return True
            return super().on_touch_down(touch)

    class CloudView(FrameImage):
        def __init__(self,**kw):
            super().__init__(**kw);self.gesture='rotate';self.renderer=CloudRenderer();self.packet=None;self.touches={};self.start={};self.pinch=None;self.pick_side=0;self.on_pick=None
            self.trigger=Clock.create_trigger(self.render,1/30);self.bind(size=lambda *_:self.trigger())
        def set_packet(self,packet,reset=False):
            self.packet=packet
            if reset:self.renderer.reset(packet)
            self.trigger()
        def render(self,*_):
            if self.width<10 or self.height<10:return
            if self.parent is None:return
            w,h=int(min(900,max(160,self.width))),int(min(600,max(120,self.height)))
            self.show(self.renderer.draw(self.packet,w,h))
        def on_touch_down(self,touch):
            if not self.collide_point(*touch.pos):return super().on_touch_down(touch)
            if 'button' in touch.profile and touch.button in ('scrollup','scrolldown'):
                self.renderer.zoom=float(np.clip(self.renderer.zoom*(1.12 if touch.button=='scrollup' else 1/1.12),.35,6));self.trigger();return True
            self.renderer.interacting=True;touch.grab(self);self.touches[touch.uid]=np.array(touch.pos,dtype=float);self.start[touch.uid]=(np.array(touch.pos,dtype=float),time.monotonic())
            if len(self.touches)==2:
                a,b=list(self.touches.values());self.pinch=(max(1,np.linalg.norm(a-b)),(a+b)/2,self.renderer.zoom,self.renderer.pan.copy(),math.atan2(b[1]-a[1],b[0]-a[0]),self.renderer.roll)
            return True
        def on_touch_move(self,touch):
            if touch.uid not in self.touches:return super().on_touch_move(touch)
            if touch.grab_current not in (None,self):return False
            old=self.touches.get(touch.uid,np.array(touch.pos));new=np.array(touch.pos,dtype=float);self.touches[touch.uid]=new
            if len(self.touches)==1:
                delta=new-old
                if self.gesture=='pan':
                    self.renderer.pan+=np.array([delta[0]/max(1,self.width),-delta[1]/max(1,self.height)])
                elif self.gesture=='depth':
                    self.renderer.perspective=True
                    self.renderer.view_translation[2]=float(np.clip(self.renderer.view_translation[2]+delta[1],-250.,800.))
                else:
                    self.renderer.yaw=(self.renderer.yaw+delta[0]*.45)%360
                    self.renderer.pitch=float(np.clip(self.renderer.pitch-delta[1]*.45,-89,89))
            elif len(self.touches)==2 and self.pinch:
                a,b=list(self.touches.values());dist,center,zoom,pan,angle,roll=self.pinch
                self.renderer.zoom=float(np.clip(zoom*np.linalg.norm(a-b)/dist,.35,6))
                self.renderer.roll=roll+math.degrees(math.atan2(b[1]-a[1],b[0]-a[0])-angle)
                movement=(a+b)/2-center;self.renderer.pan=pan+np.array([movement[0]/max(1,self.width),-movement[1]/max(1,self.height)])
            self.trigger();return True
        def on_touch_up(self,touch):
            if touch.uid not in self.touches:return super().on_touch_up(touch)
            if touch.grab_current not in (None,self):return False
            origin,started=self.start.get(touch.uid,(np.array(touch.pos),0))
            was_single=len(self.touches)==1 and self.pinch is None
            if was_single and np.linalg.norm(np.array(touch.pos)-origin)<dp(10) and time.monotonic()-started<.55 and self.renderer.show_landmarks:
                # Convert touch to the actual letterboxed texture, not widget bounds.
                tw,th=self.renderer.last_size;factor=min(self.width/tw,self.height/th);ox=self.x+(self.width-tw*factor)/2;oy=self.y+(self.height-th*factor)/2
                p=np.array([(touch.x-ox)/factor,th-(touch.y-oy)/factor])
                hits=self.renderer.hit_points
                if hits:
                    name=min(hits,key=lambda k:np.linalg.norm(np.array(hits[k])-p))
                    if np.linalg.norm(np.array(hits[name])-p)<max(20,dp(25)/factor):
                        if self.on_pick:self.on_pick(name,self.pick_side)
                        self.pick_side=1-self.pick_side
            self.touches.pop(touch.uid,None);self.start.pop(touch.uid,None)
            if not self.touches:self.pinch=None;self.renderer.interacting=False;self.trigger()
            touch.ungrab(self);return True

    class FaceApp(App):
        def build(self):
            settings=read_settings()
            self.engine=Engine(args.demo,args.ir_dot,args.ir_flood);self.engine.start();self.patient='Anonymous';self.pose_mode=args.pose;self.review_packet=None;self.live_review=False;self.saving=False;self.capturing=False;self.capture_progress='';self.mark_name=None
            self.smoothing_controls=[];self._syncing_smoothing=False
            self.smooth_enabled=True;self.smooth_window=5;self.smooth_method='mean';self.debug_info=False;self.capture_cloud=False;self.scan_number=None
            self.mode='summary';self.result_scope='scan';self.selected_pair='CHEILION';self.include_depth=False;self.measure_pair=('PRONASALE','CHEILION_R');self.review_photo=False;self.session_packets={};self.scan_views={};self.review_view_key=None
            self.exporting=False;self.export_progress='';self.export_threads=[];self.panel_key=None
            self.frame_event=None;self.frame_index=0;self.burst_packet=None;self.control_token=None;self.frame_loading=False;self.footer_buttons={}
            self.live_asi_enabled=True;self.capture_dense=False;self.capture_points=False;self.smooth_enabled=settings.get('smooth_enabled',True);self.smooth_window=max(1,min(90,int(settings.get('smooth_window',5))));self.smooth_method=settings.get('smooth_method','mean') if settings.get('smooth_method','mean') in ('mean','median') else 'mean'
            self.last_seq=-1;self.status_token=None;self.photo_token=None;self.manager=ScreenManager(transition=NoTransition())
            self.build_home();self.build_capture();self.build_library();self.build_review()
            Clock.schedule_interval(self.tick,.05);Window.bind(on_keyboard=self.key)
            if args.screenshot_dir:Clock.schedule_once(self.screenshots,1.5)
            return self.manager
        def topbar(self,title,back='home'):
            bar=BoxLayout(size_hint_y=None,height=dp(56),spacing=dp(10),padding=[dp(12),dp(6)])
            bar.add_widget(button('< Home' if back=='home' else '< Back',lambda:self.go(back),height=44,size_hint_x=None,width=dp(86)))
            bar.add_widget(label(title,18,bold=True));bar.add_widget(button('Exit',self.request_exit,height=44,size_hint_x=None,width=dp(58)));return bar
        def build_home(self):
            screen=Screen(name='home');root=BoxLayout(orientation='vertical',padding=[dp(32),dp(24)],spacing=dp(12));screen.add_widget(root)
            title=BoxLayout(size_hint_y=None,height=dp(64));title.add_widget(label('FaceASI',32,bold=True));title.add_widget(label(f'MEASURE  /  v{VERSION}',13,color=TEAL,halign='right'));title.add_widget(button('Exit',self.request_exit,height=44,size_hint_x=None,width=dp(70)));root.add_widget(title)
            root.add_widget(label('Neutral / Smiling / Blinking · live 3D ASI · scan review.',17,color=MUTED,size_hint_y=None,height=dp(32)))
            row=BoxLayout(spacing=dp(16))
            row.add_widget(button('Capture\n\nPosition at 30-40 cm',self.open_capture,tint=(.08,.43,.39,1),height=160,size_hint_y=1,font_size=sp(24)))
            row.add_widget(button('Review\n\nExplore your saved scans',lambda:self.go('library'),height=160,size_hint_y=1,font_size=sp(22)))
            root.add_widget(row)
            idrow=BoxLayout(size_hint_y=None,height=dp(42),spacing=dp(12));idrow.add_widget(label('Subject label',13,color=MUTED,size_hint_x=None,width=dp(105)))
            self.id_input=TextInput(text='Subject01',multiline=False,font_size=sp(16),background_color=CARD,foreground_color=TEXT,cursor_color=TEAL,padding=[dp(12),dp(10)])
            idrow.add_widget(self.id_input);root.add_widget(idrow)
            self.home_status=label('Starting OAK camera…',13,color=MUTED,size_hint_y=None,height=dp(28));root.add_widget(self.home_status)
            self.manager.add_widget(screen)
        def build_capture(self):
            # Controls occupy separate bars: nothing covers the camera/cloud.
            screen=Screen(name='capture');root=BoxLayout(orientation='vertical',spacing=dp(3))
            top=BoxLayout(size_hint_y=None,height=dp(44),spacing=dp(4))
            top.add_widget(button('< Home',lambda:self.go('home'),height=44,size_hint_x=None,width=dp(80)))
            self.distance_title=label('Finding face…',15,bold=True);top.add_widget(self.distance_title)
            self.capture_toggle=button('Point cloud',self.toggle_capture_view,height=44,size_hint_x=None,width=dp(120));top.add_widget(self.capture_toggle)
            top.add_widget(button('Face points',self.capture_points_menu,height=44,size_hint_x=None,width=dp(100)));top.add_widget(button('Settings',self.capture_settings,height=44,size_hint_x=None,width=dp(95)));root.add_widget(top)
            self.capture_stage=BoxLayout();self.feed=FrameImage();self.capture_cloud_view=CloudView()
            self.capture_cloud_view.renderer.show_landmarks=False;self.capture_cloud_view.renderer.measure_pair=()
            self.capture_stage.add_widget(self.feed);root.add_widget(self.capture_stage)
            self.live_values=label('',13,markup=True,size_hint_y=None,height=dp(92),halign='center');root.add_widget(self.live_values)
            self.guide_text=label('',12,size_hint_y=None,height=dp(24));root.add_widget(self.guide_text)
            bottom=BoxLayout(size_hint_y=None,height=dp(52),spacing=dp(4));self.pose_buttons={}
            for name in CAPTURE_MODES:
                display='Normal' if name=='Neutral' else name
                widget=button(display,lambda mode=name:self.select_pose(mode),height=52)
                self.pose_buttons[name]=widget;bottom.add_widget(widget)
            self.capture_button=button(f'Capture {args.frames}F',self.take_capture,tint=TEAL,height=52);bottom.add_widget(self.capture_button)
            self.capture_button.disabled=True;root.add_widget(bottom)
            # Kept for legacy status consumers, deliberately not on capture screen.
            self.pose_hint=label('');self.capture_numbers=label('');self.update_pose_controls()
            screen.add_widget(root);self.manager.add_widget(screen)
        def toggle_capture_view(self):
            self.capture_cloud=not self.capture_cloud
            self.capture_stage.clear_widgets();self.capture_stage.add_widget(self.capture_cloud_view if self.capture_cloud else self.feed)
            self.capture_toggle.text='Camera' if self.capture_cloud else 'Point cloud';self.last_seq=-1
        def capture_settings(self):
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(8))
            popup=Popup(title='Capture and export settings',content=content,size_hint=(.95,.88))
            content.add_widget(label('CSV export folder (local, USB, or your synced folder)',13,size_hint_y=None,height=dp(35)))
            field=TextInput(text=str(configured_scans_dir() or ''),multiline=False,size_hint_y=None,height=dp(46))
            content.add_widget(field)
            content.add_widget(button('Browse results folder',lambda:self.export_folder_popup(on_selected=lambda p:setattr(field,'text',str(p))),height=42))
            self.add_smoothing_controls(content)
            content.add_widget(button('Calibrate relaxed face / eyes open',self.calibrate_face,height=44))
            row=BoxLayout(size_hint_y=None,height=dp(38));live=CheckBox(active=self.live_asi_enabled,size_hint_x=None,width=dp(42))
            def set_live(_,value):
                self.live_asi_enabled=value;save_settings(v53_live_asi_enabled=value);self.last_seq=-1
                self.live_values.height=dp(92) if value else 0;self.live_values.opacity=1 if value else 0
            live.bind(active=set_live);row.add_widget(live);row.add_widget(label('Show live smoothed ASI',13));content.add_widget(row)
            row=BoxLayout(size_hint_y=None,height=dp(38));dense=CheckBox(active=self.capture_dense,size_hint_x=None,width=dp(42))
            dense.bind(active=lambda _,v:self.set_capture_dense(v));row.add_widget(dense);row.add_widget(label('Detailed face points (eyes / brows / lips)',13));content.add_widget(row)
            content.add_widget(label(f'Each completed capture automatically exports CSV to:\n{AUTO_CSV_DIR}\nThe folder above is for additional manual exports.',13))
            def apply():
                if field.text.strip():
                    target=Path(field.text.strip()).expanduser()
                    if not target.is_absolute() or not target.is_dir():
                        notice('Folder not found','Choose an existing folder using Browse results folder.');return
                    try:remember_export_folder(target)
                    except (OSError,ValueError) as exc:notice('Cannot use folder',str(exc));return
                popup.dismiss()
            content.add_widget(button('Save settings',apply,height=48,tint=TEAL));popup.open()
        def add_smoothing_controls(self,parent):
            parent.add_widget(button('Smoothing settings / averaging window',self.smoothing_popup,height=48,font_size=sp(13)))
        def smoothing_popup(self):
            from kivy.uix.gridlayout import GridLayout
            content=BoxLayout(orientation='vertical',padding=dp(10),spacing=dp(8))
            popup=Popup(title='Stabilize live and frame ASI',content=content,size_hint=(.92,.91))
            status=label('',15,size_hint_y=None,height=dp(44));content.add_widget(status)
            def refresh():status.text=('Moving average' if self.smooth_method=='mean' else 'Median smoothing')+f' | {self.smooth_window} frames'+(' | ON' if self.smooth_enabled else ' | OFF')
            def choose(n):self.set_smoothing(window=n,enabled=True);refresh()
            grid=GridLayout(cols=4,spacing=dp(8),size_hint_y=None,height=dp(166))
            for n in (1,3,5,10,12,15,20,30,45,60,75,90):grid.add_widget(button(str(n)+' frames',lambda n=n:choose(n),height=50))
            content.add_widget(grid)
            row=BoxLayout(size_hint_y=None,height=dp(48),spacing=dp(8))
            def method(m):self.set_smoothing(method=m,enabled=True);refresh()
            row.add_widget(button('Moving average',lambda:method('mean'),height=48))
            row.add_widget(button('Median',lambda:method('median'),height=48))
            row.add_widget(button('Turn off',lambda:(self.set_smoothing(enabled=False),refresh()),height=48));content.add_widget(row)
            content.add_widget(label('Start with 5 frames for expression changes.\nMore frames reduce jitter but add delay.\nOnly past/current frames with matching valid pairs are averaged.\nWhole-scan ASI is a separate recording summary.',13))
            content.add_widget(button('Done',popup.dismiss,height=48,tint=TEAL));refresh();popup.open()
        def capture_points_menu(self):
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(10))
            popup=Popup(title='Points and movement checks',content=content,size_hint=(.85,.92))
            def choose(kind):
                self.capture_points=kind=='key';self.set_capture_dense(kind=='all')
                self.capture_cloud_view.renderer.show_landmarks=self.capture_points
                self.last_seq=-1;popup.dismiss()
            for kind,title in [('off','Camera / surface only'),('key','Key measurement landmarks'),('all','Detailed face points: eyes, brows, lips')]:
                content.add_widget(button(title,lambda k=kind:choose(k),height=56))
            content.add_widget(button('Check eyelid / eyebrow movement',lambda:(popup.dismiss(),self.upper_checks()),height=44))
            content.add_widget(button('Calibrate relaxed face',lambda:(popup.dismiss(),self.calibrate_face()),height=44))
            popup.open()
        def calibrate_face(self):
            if self.capturing or self.saving:notice('Finish recording first','Calibrate before starting a new recording.');return
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(8))
            popup=Popup(title='Calibrate this person',content=content,size_hint=(.85,.65))
            content.add_widget(label('Relax your mouth and eyebrows. Keep eyes open.\nFace the camera. Hold this while 15 new frames are collected.\nCalibration affects expression checks only, not 3D ASI.',14))
            def begin():
                self.engine.calibrate_expression();self.engine.active.set();popup.dismiss()
            content.add_widget(button('Start neutral calibration',begin,height=50,tint=TEAL))
            content.add_widget(button('Cancel',popup.dismiss,height=44));popup.open()
        def upper_checks_text(self,packet):
            if packet is None or (self.manager.current=='capture' and packet_age(packet)>STALE_SECONDS):return 'No fresh face data.'
            e=packet.expression;f=e.get('features',{})
            lines=[expression_line(packet),e.get('note',''),'2D shape checks; separate from stereo-depth ASI.']
            if not f:lines.append('Detailed eye landmarks were not captured in this frame.')
            else:
                lines.extend([f"Eye gap / eye width: left {f['eye_L']:.3f}, right {f['eye_R']:.3f}",
                    f"Eye opening asymmetry: {f['eye_difference']:.1f}%",
                    f"Brow-to-eye gap / eye width: left {f['brow_L']:.3f}, right {f['brow_R']:.3f}",
                    f"Brow height asymmetry: {f['brow_difference']:.1f}%"])
                b=e.get('baseline',{})
                if b:lines.append(f"Brow change from relaxed: left {f['brow_L']-b['brow_L']:+.3f}, right {f['brow_R']-b['brow_R']:+.3f}")
            lines.extend(['Both eyes can close or both brows rise while left/right asymmetry stays low.',
                'Primary upper 3D ASI uses brow and eye-corner positions. Eyelid opening is checked separately here.',
                'Expression labels use threshold rules, not a trained expression classifier. A correct label does not prove correct depth or ASI.'])
            return '\n\n'.join(lines)
        def upper_checks(self):
            capture=self.manager.current=='capture'
            def selected():return self.engine.snapshot() if capture else self.review_packet
            popup=notice('Eye opening / eyebrow movement checks',self.upper_checks_text(selected()))
            message=next(w for w in popup.content.walk() if type(w) is Label)
            if capture:
                event=Clock.schedule_interval(lambda *_:setattr(message,'text',self.upper_checks_text(selected())),.15)
                popup.bind(on_dismiss=lambda *_:event.cancel())
            return popup
        def set_capture_dense(self,value):
            self.capture_dense=bool(value);self.capture_cloud_view.renderer.show_dense=self.capture_dense;self.last_seq=-1
        def set_smoothing(self,enabled=None,window=None,method=None):
            if self._syncing_smoothing:return
            if enabled is not None:self.smooth_enabled=enabled
            if window is not None:self.smooth_window=max(1,min(90,int(window)))
            if method is not None:self.smooth_method=method
            self._syncing_smoothing=True
            try:
                self.smoothing_controls=[refs for refs in self.smoothing_controls if all(ref() is not None for ref in refs)]
                for refs in self.smoothing_controls:
                    check,window,method_control=(ref() for ref in refs)
                    check.active=self.smooth_enabled;window.text=str(self.smooth_window);method_control.text=self.smooth_method
            finally:self._syncing_smoothing=False
            save_settings(smooth_enabled=self.smooth_enabled,smooth_window=self.smooth_window,smooth_method=self.smooth_method)
            self.status_token=None;self.last_seq=-1
            if hasattr(self,'review_packet'):self.update_panel()
        def review_history(self):
            packet=self.review_packet
            if packet is None:return []
            if self.live_review:
                with self.engine.lock:frames=list(self.engine.metric_history)
                return [f for f in frames if f.seq<=packet.seq and f.timestamp<=packet.timestamp] or [packet]
            frames=self.cached_frames()
            return frames[:self.frame_index+1] if frames else [packet]
        def displayed_value(self,getter):
            return smooth_value(self.review_history(),getter,self.smooth_window if self.smooth_enabled else 1,self.smooth_method)
        def toggle_debug(self,active):
            self.debug_info=active;self.panel_key=None;self.rebuild_panel();self.update_panel()
        def toggle_render_option(self,key,active):
            setattr(self.cloud_view.renderer,key,active);self.photo_token=None;self.status_token=None;self.cloud_view.trigger();self.update_panel()
        def debug_snapshot(self):
            if not args.save_photos:
                notice('Snapshot disabled','Restart with --save-photos to explicitly enable debug images.');return
            destination=DATA_ROOT.parent/'debug';destination.mkdir(parents=True,exist_ok=True)
            import uuid
            path=destination/('debug_'+uuid.uuid4().hex[:10]+'.png')
            self.viewport.export_to_png(str(path));notice('Debug snapshot saved',str(path))
        def select_pose(self,mode):
            if mode not in CAPTURE_MODES or self.saving:return
            self.pose_mode=mode;self.update_pose_controls()
        def update_pose_controls(self):
            for name,button_widget in getattr(self,'pose_buttons',{}).items():
                button_widget.tint=TEAL if name==self.pose_mode else CARD
            prompts={'Neutral':'Relax face; look straight; hold neutral pose.',
                     'Smiling':'Record neutral, smile, then return to neutral.',
                     'Blinking':'Blink during burst; fast eyelid closure may be missed.'}
            if hasattr(self,'pose_hint'):self.pose_hint.text=prompts[self.pose_mode]

        def build_library(self):
            screen=Screen(name='library');root=BoxLayout(orientation='vertical',padding=dp(10),spacing=dp(10));root.add_widget(self.topbar('Saved captures'))
            self.library_scroll=ArrowScroll(do_scroll_x=False);self.library_items=BoxLayout(orientation='vertical',spacing=dp(9),size_hint_y=None)
            self.library_items.bind(minimum_height=self.library_items.setter('height'));self.library_scroll.add_widget(self.library_items);root.add_widget(self.library_scroll)
            root.add_widget(button('New capture',self.open_capture,tint=TEAL,height=48));screen.add_widget(root);self.manager.add_widget(screen)
        def build_review(self):
            screen=Screen(name='review');root=BoxLayout(orientation='vertical',padding=dp(8),spacing=dp(6))
            header=self.topbar('Capture review');self.review_title=header.children[1]
            header.remove_widget(header.children[0]);self.review_title.font_size=sp(15);self.review_title.shorten=True
            self.live_button=button('Live 3D',self.toggle_live,height=44,size_hint_x=None,width=dp(94))
            self.export_button=button('Export CSV',self.choose_export,height=44,size_hint_x=None,width=dp(88),tint=TEAL)
            self.photo_button=button('Photo view',self.toggle_photo,height=44,size_hint_x=None,width=dp(94),font_size=sp(12),tint=TEAL)
            self.frames_button=button('Playback',self.choose_frame,height=44,size_hint_x=None,width=dp(84));self.review_header=header
            root.add_widget(header)
            modes=BoxLayout(size_hint_y=None,height=dp(42),spacing=dp(7));self.mode_buttons={}
            b=button('Scan result',lambda:self.set_mode('summary'),height=42);self.mode_buttons['summary']=b;modes.add_widget(b)
            from kivy.uix.dropdown import DropDown
            self.distance_dropdown=DropDown()
            self.distance_menu=button('Distances v',self.open_distances,height=42)
            for mode,title in [('distance','Between two landmarks'),('mouth','Nose to mouth corners'),('brow','Forehead to eyebrows')]:
                item=button(title,lambda m=mode:(self.distance_dropdown.dismiss(),self.set_mode(m)),height=52)
                self.distance_dropdown.add_widget(item)
            modes.add_widget(self.distance_menu)
            modes.add_widget(button('Other ASI formulas',self.other_asi_menu,height=42))
            modes.add_widget(button('Smoothing',self.smoothing_popup,height=42))
            root.add_widget(modes)
            body=BoxLayout(spacing=dp(10));self.viewport=FloatLayout(size_hint_x=.52);backdrop(self.viewport,(.055,.08,.115,1))
            self.cloud_view=CloudView(size_hint=(1,1),pos_hint={'x':0,'y':0});self.cloud_view.on_pick=self.picked;self.photo_view=PhotoView(size_hint=(1,1),pos_hint={'x':0,'y':0});self.viewport.add_widget(self.cloud_view)
            # Overlay controls stay above the photo/cloud and receive taps first.
            self.rotation_buttons={}
            for name,text,yaw,pitch,position in [
                ('left','←',-15,0,{'x':.015,'center_y':.5}),
                ('right','→',15,0,{'right':.985,'center_y':.5}),
                ('up','↑',0,-15,{'center_x':.5,'top':.985}),
                ('down','↓',0,15,{'center_x':.5,'y':.015})]:
                arrow=arrow_button(name,lambda y=yaw,p=pitch:self.rotate_scan(y,p),
                    height=50,size_hint=(None,None),width=dp(56),
                    pos_hint=position,font_size=sp(28),bold=True,tint=TEAL)
                self.rotation_buttons[name]=arrow;self.viewport.add_widget(arrow)
            body.add_widget(self.viewport);self.side_scroll=ArrowScroll(size_hint_x=.48,do_scroll_x=False)
            self.side_panel=BoxLayout(orientation='vertical',spacing=dp(5),padding=[dp(10),dp(5)],size_hint_y=None)
            self.side_panel.bind(minimum_height=self.side_panel.setter('height'));self.side_scroll.add_widget(self.side_panel);body.add_widget(self.side_scroll);root.add_widget(body)
            self.review_layout=root
            self.frame_bar=BoxLayout(size_hint_y=None,height=dp(44),spacing=dp(8))
            self.previous_frame_button=button('Previous',lambda:self.step_frame(-1),height=44)
            self.inline_play_button=button('Play',self.toggle_playback,height=44,tint=TEAL)
            self.next_frame_button=button('Next',lambda:self.step_frame(1),height=44)
            self.frame_readout=label('Frame',13,halign='center')
            for item in (self.previous_frame_button,self.frame_readout,self.inline_play_button,self.next_frame_button):self.frame_bar.add_widget(item)
            self.footer=BoxLayout(size_hint_y=None,height=dp(46),spacing=dp(7));root.add_widget(self.footer)
            screen.add_widget(root);self.manager.add_widget(screen);self.set_mode('summary')
        def go(self,name):
            if getattr(self,'saving',False) and name!='review':return
            self.stop_playback()
            self.mark_name=None
            if hasattr(self,'photo_view'):self.photo_view.on_mark=None
            self.manager.current=name
            if name=='library':self.engine.active.clear();self.live_review=False;self.refresh_library()
            elif name=='home':self.engine.active.clear();self.live_review=False
            elif name=='review':
                if self.live_review:self.engine.active.set()
                else:self.engine.active.clear()
            elif name=='capture':
                with self.engine.lock:self.engine.metric_history.clear()
                self.engine.active.set();self.last_seq=-1
        def request_exit(self):
            content=BoxLayout(orientation='vertical',padding=dp(16),spacing=dp(12))
            content.add_widget(label('Close FaceASI and release the camera?',18))
            pop=Popup(title='Exit application',content=content,size_hint=(.7,.55))
            content.add_widget(button('Exit app',lambda:(pop.dismiss(),self.stop()),tint=WARN,height=54))
            content.add_widget(button('Cancel',pop.dismiss,height=48));pop.open()
        def can_mark(self):
            p=self.review_packet
            return bool(p is not None and not self.live_review and not self.saving and not self.exporting and p.rgb is not None and p.depth is not None)
        def refresh_controls(self):
            if not hasattr(self,'footer'):return
            p=self.review_packet;r=self.cloud_view.renderer
            root=self.burst_packet;has_frames=bool(root and root.frames and not self.live_review and not self.frame_loading)
            key=(p is not None,p is not None and p.rgb is not None,has_frames,self.engine.error is None,p is not None and bool(p.landmarks),self.live_review,self.review_photo,self.saving,self.exporting,r.show_landmarks,r.show_cloud,r.color_mode,self.cloud_view.gesture)
            can_export=bool(p is not None and p.path is not None and Path(p.path).is_dir() and not self.saving and not self.live_review and not self.exporting)
            photo_available=self.review_photo or bool(p is not None and (p.rgb is not None or (not self.live_review and root is not None and root.rgb is not None)))
            key=key+(can_export,photo_available)
            self.update_frame_readout()
            if key==self.control_token:return
            self.control_token=key
            self.photo_button.text='3D cloud' if self.review_photo else 'Photo view'
            if has_frames and self.frame_bar.parent is None:self.review_layout.add_widget(self.frame_bar,index=1)
            if not has_frames and self.frame_bar.parent:self.review_layout.remove_widget(self.frame_bar)
            self.update_frame_readout()
            if photo_available and self.photo_button.parent is None:self.review_header.add_widget(self.photo_button)
            if not photo_available and self.photo_button.parent:self.review_header.remove_widget(self.photo_button)
            if can_export and self.export_button.parent is None:self.review_header.add_widget(self.export_button)
            if not can_export and self.export_button.parent:self.review_header.remove_widget(self.export_button)
            if has_frames and self.frames_button.parent is None:self.review_header.add_widget(self.frames_button)
            if not has_frames and self.frames_button.parent:self.review_header.remove_widget(self.frames_button)
            if self.engine.error is not None and self.live_button.parent:self.review_header.remove_widget(self.live_button)
            # Live cloud is now on Capture, not Review.
            self.footer.clear_widgets();self.footer_buttons={}
            if p is None:return
            def add(name,text,fn,tint=CARD):
                b=button(text,fn,height=46,tint=tint);self.footer.add_widget(b);self.footer_buttons[name]=b;return b
            self.view_button=add('view','View options',self.view_options)
            if not self.review_photo:
                for mode,text in [('rotate','Rotate'),('pan','Move view')]:
                    add(mode,text,lambda m=mode:self.select_gesture(m),TEAL if self.cloud_view.gesture==mode else CARD)
                add('zoom_out','Zoom -',lambda:self.zoom(1/1.25));add('zoom_in','Zoom +',lambda:self.zoom(1.25))
            elif self.review_photo:add('cloud','3D cloud',self.toggle_photo,TEAL)
        def select_gesture(self,mode):
            self.cloud_view.gesture=mode;self.control_token=None;self.refresh_controls();self.cloud_view.trigger()
        def view_options(self):
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(7));popup=Popup(title='View controls',content=content,size_hint=(.8,.9))
            scroll=ArrowScroll(do_scroll_x=False);items=BoxLayout(orientation='vertical',spacing=dp(7),size_hint_y=None);items.bind(minimum_height=items.setter('height'));scroll.add_widget(items);content.add_widget(scroll)
            def action(title,fn):
                def run():fn();popup.dismiss()
                items.add_widget(button(title,run,height=45))
            p=self.review_packet;r=self.cloud_view.renderer
            if p and p.rgb is not None:action('3D cloud' if self.review_photo else 'Photo with measurements',self.toggle_photo)
            if not self.review_photo:
                action('Reset front view',self.reset_view)
                action('Move model closer / farther',lambda:self.select_gesture('depth'))
                for dy,dpitch,title in [(-15,0,'Rotate left 15 degrees'),(15,0,'Rotate right 15 degrees'),(0,-15,'Rotate up 15 degrees'),(0,15,'Rotate down 15 degrees')]:
                    action(title,lambda y=dy,p=dpitch:self.rotate_scan(y,p))
                for yaw,title in [(0,'Front'),(-45,'Left oblique'),(45,'Right oblique'),(90,'Side')]:
                    def setview(y=yaw):r.yaw=y;r.pitch=0;r.roll=0;self.cloud_view.trigger()
                    action(title,setview)
                if p is not None and len(p.points):
                    action('Hide point cloud' if r.show_cloud else 'Show point cloud',self.toggle_cloud)
                    for mode,title in [('depth','Depth colors'),('rgb','Point RGB colors'),('solid','Single cyan')]:
                        def color(m=mode):r.color_mode=m;self.control_token=None;self.cloud_view.trigger();self.refresh_controls()
                        action(title,color)
            action('Close',lambda:None);popup.open()
        def toggle_cloud(self):
            r=self.cloud_view.renderer;r.show_cloud=not r.show_cloud;self.control_token=None;self.cloud_view.trigger();self.refresh_controls()
        def stop_playback(self):
            if getattr(self,'frame_event',None):self.frame_event.cancel();self.frame_event=None
            if hasattr(self,'play_button'):self.play_button.text='Play frames'
            if hasattr(self,'inline_play_button'):self.inline_play_button.text='Play'
        def attach_representative_photo(self,root,frames):
            # Compatibility with captures made before individual frame photos were saved.
            if root.rgb is None or not 0<=root.selected_frame<len(frames):return
            frame=frames[root.selected_frame]
            if frame.rgb is None and frame.seq==root.seq and abs(frame.timestamp-root.timestamp)<.001:
                frame.rgb=root.rgb
        def cached_frames(self):
            root=self.burst_packet
            if root and root.frames and all(isinstance(f,Packet) for f in root.frames):
                self.attach_representative_photo(root,root.frames);return root.frames
            return None
        def update_frame_readout(self):
            root=self.burst_packet
            if not root or not root.frames:return
            missing=self.review_photo and self.review_packet is not None and self.review_packet.rgb is None
            self.frame_readout.text=f'{self.frame_index+1}/{len(root.frames)}'+(' · no photo' if missing else '')
        def step_frame(self,delta):
            self.stop_playback()
            if not self.cached_frames():self.ensure_frames(lambda:self.step_frame(delta));return
            self.show_frame(self.frame_index+delta)
        def ensure_frames(self,callback):
            root=self.burst_packet
            if not root or not root.frames:return
            if self.cached_frames():callback();return
            if self.frame_loading:return
            self.frame_loading=True;self.control_token=None;self.refresh_controls();self.review_title.text='Loading independent frame geometry…'
            def load():
                try:
                    frames=[f if isinstance(f,Packet) else load_packet(f) for f in root.frames]
                    def ready(*_):
                        root.frames=frames;self.attach_representative_photo(root,frames);self.frame_loading=False;self.control_token=None;self.refresh_controls()
                        if self.burst_packet is root and not self.live_review:callback()
                    Clock.schedule_once(ready,0)
                except Exception as exc:
                    message=str(exc)
                    def failed(*_):self.frame_loading=False;self.control_token=None;self.refresh_controls();notice('Frames could not load',message)
                    Clock.schedule_once(failed,0)
            threading.Thread(target=load,daemon=True).start()
        def show_frame(self,index):
            frames=self.cached_frames()
            if not frames:return
            self.frame_index=index%len(frames);self.load_review(frames[self.frame_index],reset=False)
            self.review_title.text=('Photo' if self.review_photo else 'Frame')+f' · {self.frame_index+1}/{len(frames)}'+(' · no photo' if self.review_photo and self.review_packet.rgb is None else '')
            self.update_frame_readout()
            if hasattr(self,'frame_slider') and not getattr(self,'slider_lock',False):
                self.slider_lock=True;self.frame_slider.value=self.frame_index;self.slider_lock=False
            if hasattr(self,'frame_label'):self.frame_label.text=f'{self.frame_index+1}/{len(frames)}'
        def toggle_playback(self):
            if self.frame_event:self.stop_playback();return
            if not self.cached_frames():self.ensure_frames(self.toggle_playback);return
            self.update_view()
            self.frame_event=Clock.schedule_interval(lambda *_:self.show_frame(self.frame_index+1),.20)
            if hasattr(self,'play_button'):self.play_button.text='Pause frames'
            self.inline_play_button.text='Pause'
        def choose_frame(self):
            self.ensure_frames(self.frame_controls)
        def frame_controls(self):
            frames=self.cached_frames()
            if not frames or self.live_review:return
            content=BoxLayout(orientation='vertical',padding=dp(14),spacing=dp(10));popup=Popup(title='Independent frames — one pose',content=content,size_hint=(.85,.72),auto_dismiss=False)
            self.frame_label=label(f'{self.frame_index+1}/{len(frames)}',18,height=dp(35),size_hint_y=None);content.add_widget(self.frame_label)
            self.frame_slider=Slider(min=0,max=len(frames)-1,value=self.frame_index,step=1,height=dp(48),size_hint_y=None)
            self.slider_lock=False
            def slide(_,value):
                if not self.slider_lock:self.stop_playback();self.show_frame(int(value))
            self.frame_slider.bind(value=slide);content.add_widget(self.frame_slider)
            row=BoxLayout(size_hint_y=None,height=dp(46),spacing=dp(8))
            row.add_widget(button('Previous',lambda:(self.stop_playback(),self.show_frame(self.frame_index-1)),height=46))
            self.play_button=button('Pause frames' if self.frame_event else 'Play frames',self.toggle_playback,height=46,tint=TEAL);row.add_widget(self.play_button)
            row.add_widget(button('Next',lambda:(self.stop_playback(),self.show_frame(self.frame_index+1)),height=46));content.add_widget(row)
            content.add_widget(button('Selected distance + nose Z graphs',lambda:(popup.dismiss(),self.selected_trace()),height=46))
            content.add_widget(button('ASI raw + filtered graphs',lambda:(popup.dismiss(),self.show_asi_trace()),height=46))
            content.add_widget(button('ASI / mouth repeatability',self.show_repeatability,height=46))
            content.add_widget(label('Photo/cloud, landmarks and numbers use the same frame.\nScene origin stays fixed so motion remains visible.',13,height=dp(50),size_hint_y=None))
            content.add_widget(button('Back to review',popup.dismiss,height=46))
            popup.open()
        def pair_summary(self):
            frames=self.cached_frames()
            if not frames or self.live_review:return ''
            a,b=self.measure_pair;stats=pair_statistics(frames,a,b)
            return f"{stats['valid']}/{stats['total']} frames · mean {fmt(stats['mean'])} mm\nSD {fmt(stats['sd'])} · range {fmt(stats['range'])} mm"
        def selected_trace(self):
            self.ensure_frames(self.show_selected_trace)
        def show_selected_trace(self):
            self.stop_playback();frames=self.cached_frames()
            if not frames:return
            a,b=self.measure_pair;stats=pair_statistics(frames,a,b)
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(8));popup=Popup(title=a+' → '+b,content=content,size_hint=(.95,.95))
            content.add_widget(label(f"Valid {stats['valid']}/{stats['total']} · mean {fmt(stats['mean'])} mm · SD {fmt(stats['sd'])} mm · range {fmt(stats['range'])} mm · CV {fmt(stats['cv'])}%",13,height=dp(35),size_hint_y=None))
            filtered=[smooth_value(frames[:i+1],lambda p:point_distance(p.landmarks,a,b),self.smooth_window if self.smooth_enabled else 1,self.smooth_method) for i in range(len(frames))]
            content.add_widget(TracePlot(values=stats['values'],secondary=filtered,title='A-B distance / mm: orange raw, teal filtered'))
            noses=[f.landmarks.get('PRONASALE') for f in frames];z=[float(p[2]) if finite_point(p) else None for p in noses]
            content.add_widget(TracePlot(values=z,title='Pronasale camera Z / mm'))
            valid_z=np.array([n for n in z if n is not None],float)
            mean_z=float(valid_z.mean()) if len(valid_z) else None;sd_z=float(valid_z.std(ddof=1)) if len(valid_z)>1 else None
            content.add_widget(label(f'Nose Z mean {fmt(mean_z)} mm · SD {fmt(sd_z)} mm · valid {len(valid_z)}/{len(z)}',12,height=dp(25),size_hint_y=None))
            root=self.burst_packet
            if root.path:
                with (Path(root.path)/f'selected_{a}_{b}_frames.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['frame','distance_mm','nose_camera_Z_mm'])
                    for i,(d,n) in enumerate(zip(stats['values'],z)):w.writerow([i+1,d,n])
            content.add_widget(button('Close',popup.dismiss,height=44));popup.open()

        def show_asi_trace(self):
            self.stop_playback();frames=self.cached_frames()
            if not frames:return
            content=BoxLayout(orientation='vertical',padding=dp(10),spacing=dp(5))
            popup=Popup(title='ASI: orange raw / teal filtered',content=content,size_hint=(.96,.96))
            summaries=[asi_display_summary(frames[:i+1],self.smooth_window,self.smooth_enabled,self.smooth_method) for i in range(len(frames))]
            for region in ('upper','lower','overall'):
                content.add_widget(TracePlot(values=[f.scores['asi'][region] for f in frames],secondary=[a['display'][region] for a in summaries],title=region.upper()+' ASI (%)'))
            content.add_widget(button('Back to review',popup.dismiss,height=42));popup.open()

        def show_repeatability(self):
            root=getattr(self,'burst_packet',None)
            if root is None or not root.frames:return
            if root.path and (Path(root.path)/'burst_summary.csv').exists():
                with (Path(root.path)/'burst_summary.csv').open() as f:rows=list(csv.DictReader(f))
                lines=[]
                for r in rows:
                    mean=float(r['mean']) if r['mean'] else None;sd=float(r['sd']) if r['sd'] else None
                    lines.append(f"{r['metric']}\nmean {fmt(mean)} · SD {fmt(sd)} · valid {r['valid_frames']}/{r['total_frames']}")
            else:
                rows=burst_statistics(self.cached_frames() or []);lines=[f"{r['metric']}\nmean {fmt(r['mean'])} · SD {fmt(r['sd'])} · valid {r['valid_frames']}/{r['total_frames']}" for r in rows]
            notice('Within-pose repeatability','\n\n'.join(lines)+'\n\nSD describes frame variation, not absolute accuracy. Displayed geometry and scores belong to one actual frame. Summary is from the original automatic landmarks, before manual corrections.')
        def show_quality(self):
            packet=self.review_packet
            if packet is None:return
            tally=landmark_tally(packet.landmarks,packet.landmark_pixels,packet.quality)
            lines=[f"Auto IDs: {tally['total']}; 2D found: {tally['detected_2d']}; 3D valid: {tally['valid_3d']}; strict depth good: {tally['good_depth']}.",
                   'A missing point prevents only the measurements requiring it.\nCaution points are measured, but may be unreliable.']
            for name in list(LM)+['TRICHION']:
                q=packet.quality.get(name,{});present=finite_point(packet.landmarks.get(name))
                status=q.get('status') or ('available' if present else 'not detected / no depth')
                lines.append(f"{name}: {status}\nspread {fmt(q.get('depth_spread_mm'),' mm')} · pixels {q.get('valid_pixels','?')}")
            lines.append('OLD missing: '+', '.join(packet.scores['old_missing']))
            lines.append('FULL NEW missing: '+', '.join(packet.scores['new_missing']))
            notice('Measured landmark depth quality','\n\n'.join(lines))
        def three_asi_formula(self):
            packet=self.review_packet
            if packet is None:return
            asi=packet.scores['asi']
            lines=['RESEARCH-ONLY PRIMARY BILATERAL 3D ASI',
                   'For each available left/right anatomical pair in a face-local U/V/D reference:',
                   'W_i = |U_left| + |U_right|  (mm)',
                   'dH_i = ||U_left| - |U_right||  (mm)',
                   'dV_i = |V_left - V_right|  (mm)',
                   'dD_i = |D_left - D_right|  (mm)',
                   'H=100 sum(dH_i)/sum(W_i); likewise V and D.',
                   'ASI_3D = sqrt(H^2 + V^2 + D^2).',
                   'No separate midline M term is included.',
                   'UPPER: brow proxies, outer canthi, inner canthi.',
                   'LOWER: alare (nose wings), cheilions (mouth corners).',
                   'OVERALL: pooled available upper+lower pairs; not the average of 2 scores.',
                   'Left Zygoma is QC-tracked but excluded from the default 5-pair ASI.',
                   'A partially available set yields a partial-coverage score, not equivalent to a full set.',
                   'Reference: Subnasale origin, Nasion-to-Subnasale vertical; lateral axis from valid canthus/brow pairs.',
                   'Reference drift, head pose, depth noise and incorrect landmarks bias the scores.',
                   'Indices are dimensionless on a percent scale; not validated clinical severity percentages.']
            for region in ('upper','lower','overall'):
                c=asi['components'][region];coverage=asi['coverage'][region]
                lines.append(f"{region.upper()}  ASI {fmt(asi[region],'%',2)} · {coverage[0]}/{coverage[1]} pairs")
                if c:lines.append(f"   H {c['H']:.3f} · V {c['V']:.3f} · D {c['D']:.3f} · total width {c['width_mm']:.1f} mm")
            notice('3D ASI definitions / experimental', '\n\n'.join(lines))

        def anchor_formula(self):
            notice('Nose/mouth and forehead/brow','For anchor A and left/right points L,R:\ndL = ||L - A||, dR = ||R - A|| (3D mm).\nRadial asymmetry % = 100 |dR-dL| / (dR+dL).\n\nLeft/right inclination: angle between (point-anchor) and the face-down reference axis. Compare |thetaR-thetaL|.\nOpening angle: angle between the two anchor rays.\nVertical gap: |(R-L) dot face-down|.\n\nMouth anchor: pronasale. True brow anchor: manually marked trichion; automatic upper forehead point is a separate PROXY. Superciliare proxies must be checked and can be manually corrected. Radial asymmetry is a separate measurement, not the original ASI. The reference axis uses glabella, menton and inner eyes.')
        def mark_landmarks(self):
            packet=self.review_packet
            if self.live_review or packet is None or packet.rgb is None or packet.depth is None:
                notice('Frozen photo + measured depth required','Use a saved representative capture in this session, or launch with --save-photos to reopen its photo later. Individual burst frames keep geometry but no full photo/depth. Freeze live data and capture before marking.');return
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(7));popup=Popup(title='Mark a measured anatomical landmark',content=content,size_hint=(.85,.9))
            content.add_widget(label('Choose a point, then tap its exact location on the photo.\nTrichion is at the midline hairline. No depth = no XYZ.',14,height=dp(65),size_hint_y=None))
            scroll=ArrowScroll(do_scroll_x=False);items=BoxLayout(orientation='vertical',spacing=dp(5),size_hint_y=None);items.bind(minimum_height=items.setter('height'));scroll.add_widget(items);content.add_widget(scroll)
            for name in ['TRICHION','SUPERCILIARE_L','SUPERCILIARE_R',*[k for k in LM if k not in ('SUPERCILIARE_L','SUPERCILIARE_R')]]:
                def choose(n=name):
                    self.mark_name=n;self.photo_view.on_mark=self.apply_photo_mark;self.review_photo=True;self.photo_token=None;self.status_token=None;self.update_view();self.update_panel();self.review_title.text='Tap '+n+' · Home cancels';popup.dismiss()
                items.add_widget(button(name,choose,height=43))
            popup.open()
        def apply_photo_mark(self,u,v):
            packet=self.review_packet;name=self.mark_name
            if packet is None or name is None:return
            point,q=sample_landmark_depth(packet.depth,u,v,packet.intrinsics)
            if point is None:notice('No trustworthy depth at this point',q.get('status','no depth')+'\nTap again only if the anatomical location is correct.');return
            q['source']='manual photo mark + measured depth';q['status']='manual: '+q['status']
            try:persist_manual(packet,name,point,(u,v),q)
            except Exception as exc:notice('Manual correction could not save',str(exc));return
            self.mark_name=None;self.photo_view.on_mark=None;self.status_token=None;self.photo_token=None;self.cloud_view.set_packet(packet);self.update_panel();self.review_title.text='Manual '+name+' · recalculated'

        def key(self,window,key,*_):
            if key==27:
                if self.manager.current!='home':self.go('home');return True
                self.request_exit();return True
            return False
        def open_capture(self):
            patient=safe_id(self.id_input.text)
            if patient!=self.patient:
                with self.engine.lock:self.engine.expression_baseline={};self.engine.calibration_samples=None
            self.patient=patient;self.go('capture')
        def take_capture(self):
            packet=self.engine.snapshot()
            if packet is None or packet_age(packet)>STALE_SECONDS or len(packet.points)<100:return
            if self.saving:return
            import shutil
            DATA_ROOT.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(DATA_ROOT).free < max(20_000_000,args.frames*args.capture_cloud_points*110):
                notice('Storage is full','Free space in the local capture folder before recording another scan.');return
            try:self.scan_number=reserve_scan_id()
            except Exception as exc:notice('Cannot reserve scan ID',str(exc));return
            self.saving=True;self.capturing=True;self.capture_button.disabled=True;self.capture_button.text='Recording…';self.engine.active.set()
            from dataclasses import replace
            capture_window=self.smooth_window if self.smooth_enabled else 1;capture_method=self.smooth_method
            chosen_pose=self.pose_mode  # Freeze at button press: not a classifier.
            first=compact_capture_frame(packet,chosen_pose,args.capture_cloud_points)
            def save():
                try:
                    frames=[first];last=first.seq;deadline=time.monotonic()+max(30,args.frames/max(.5,min(self.engine.fps,10.))*2.5);start=first.device_meta['monotonic_s']
                    while len(frames)<args.frames and time.monotonic()<deadline and not self.engine.stop_event.is_set():
                        with self.engine.lock:candidates=[p for p in self.engine.history if p.seq>last]
                        for p in candidates:
                            frames.append(compact_capture_frame(p,chosen_pose,args.capture_cloud_points));last=p.seq
                            if not args.save_photos and len(frames)>1:
                                frames[-2].depth=None
                            if len(frames)>=args.frames:break
                        self.capture_progress=f'{chosen_pose}: {len(frames)}/{args.frames} frames · follow expression sequence';time.sleep(.02)
                    if self.engine.stop_event.is_set():return
                    if len(frames)<args.frames:raise RuntimeError(f'Only {len(frames)}/{args.frames} fresh frames received; capture again. Scan ID is reserved, not reused.')
                    for f in frames:f.device_meta.update(scan_id=f'{self.scan_number:03d}',subject_id=self.patient,elapsed_s=f.device_meta['monotonic_s']-start,smooth_window=capture_window,smooth_method=capture_method)
                    self.engine.active.clear();self.capturing=False;self.capture_progress='Saving raw frames…'
                    selected=len(frames)-1  # Open the end of the burst so trailing filters have full history.
                    packet=replace(frames[selected],pose_mode=chosen_pose,frames=frames if len(frames)>1 else None,selected_frame=selected,landmarks={k:v.copy() for k,v in frames[selected].landmarks.items()},landmark_pixels=dict(frames[selected].landmark_pixels),quality={k:dict(v) for k,v in frames[selected].quality.items()})
                    Clock.schedule_once(lambda *_:self.captured_preview(packet),0)
                    save_packet(packet,self.patient,save_photo=args.save_photos)
                    self.capture_progress='Saving result CSV…'
                    csv_path=None;export_error=None
                    try:
                        csv_path,_=auto_export_capture(packet.path)
                    except Exception as export_exc:
                        export_error=str(export_exc)
                    Clock.schedule_once(lambda *_:self.capture_done(packet,csv_path,export_error),0)
                except Exception as exc:
                    message=str(exc);Clock.schedule_once(lambda *_:self.capture_failed(message),0)
            thread=threading.Thread(target=save,daemon=False)
            self.export_threads.append(thread);thread.start()
        def captured_preview(self,packet):
            self.capturing=False;self.review_packet=packet;self.live_review=False
            self.live_button.text='Live 3D';self.live_button.tint=CARD;self.review_photo=False
            self.load_review(packet);self.review_title.text='Capture ready · saving CSV'
            if self.manager.current in ('capture','review'):self.go('review')
        def capture_done(self,packet,csv_path=None,export_error=None):
            self.session_packets[packet.path]=packet
            while len(self.session_packets)>2:self.session_packets.pop(next(iter(self.session_packets)))
            self.saving=False;self.capturing=False
            if self.review_packet is packet and not self.live_review:
                self.review_title.text=f"Scan {packet.device_meta.get('scan_id','?')} · {packet.pose_mode} · {len(packet.frames or [packet])} frames saved";self.rebuild_panel();self.update_panel()
            if csv_path is not None:
                self.last_export=csv_path
                if self.review_packet is packet:
                    self.review_title.text=f'Scan {packet.device_meta.get("scan_id","?")} · CSV saved: {csv_path.name}'
            if export_error:
                notice('Capture saved; automatic CSV export failed',
                       f'Your raw capture is safe.\n\nDestination: {AUTO_CSV_DIR}\n{export_error}\n\nUse Export CSV to retry after fixing the folder or storage issue.')
            self.refresh_controls()
        def capture_failed(self,message):
            self.saving=False;self.capturing=False;self.engine.active.set();self.capture_button.text='Capture';notice('Capture not saved',message)
        def remember_scan_view(self):
            if self.review_view_key is not None and not self.live_review:
                self.scan_views[self.review_view_key]=self.cloud_view.renderer.view_state()
                if len(self.scan_views)>100:self.scan_views.pop(next(iter(self.scan_views)))
        def load_review(self,packet,reset=True):
            if reset:
                self.remember_scan_view();self.stop_playback();self.burst_packet=None
                self.review_view_key=('path',str(Path(packet.path).resolve())) if packet.path else ('session',packet.timestamp,packet.seq)
                # A new scan always opens as an interactive 3D view.
                self.review_photo=False;self.cloud_view.gesture='rotate';self.result_scope='scan';self.mode='summary';self.panel_key=None
                self.cloud_view.renderer.mode='summary'

            self.control_token=None
            if packet.frames:self.burst_packet=packet
            self.mark_name=None;self.photo_view.on_mark=None
            if reset:self.frame_index=packet.selected_frame
            self.review_packet=packet;self.cloud_view.set_packet(packet,reset=reset)
            if reset and self.review_view_key in self.scan_views:
                self.cloud_view.renderer.restore_view(self.scan_views[self.review_view_key]);self.cloud_view.trigger()
            self.status_token=None;self.photo_token=None
            self.update_view();self.rebuild_panel();self.update_panel();self.refresh_controls()
            if reset:self.frame_index=packet.selected_frame
            if reset and packet.frames and not self.cached_frames():self.ensure_frames(lambda:self.show_frame(packet.selected_frame))
            self.review_title.text=('Live 3D' if self.live_review else f"Scan {packet.device_meta.get('scan_id','legacy')} · {packet.pose_mode}")+(f' · {len(packet.frames)}F' if packet.frames else '')+(' · DEMO' if packet.demo else '')
        def toggle_live(self):
            self.remember_scan_view()
            self.stop_playback();self.control_token=None
            self.live_review=not self.live_review;self.live_button.text='Freeze live' if self.live_review else 'Live 3D';self.live_button.tint=TEAL if self.live_review else CARD
            if self.live_review:
                self.review_view_key=None
                self.review_photo=False;self.update_view();self.engine.active.set();self.last_seq=-1;self.cloud_view.renderer.reset()
            else:
                self.engine.active.clear()
                if self.review_packet:self.review_title.text='Frozen live preview · unsaved'
            self.rebuild_panel();self.refresh_controls()
        def toggle_photo(self):
            packet=self.review_packet;self.stop_playback()
            if packet is None:return
            self.cached_frames()  # Restore only the matching legacy representative frame photo.
            self.review_photo=not self.review_photo;self.control_token=None;self.update_view();self.photo_token=None;self.status_token=None;self.rebuild_panel();self.update_panel();self.update_frame_readout()
            if self.burst_packet and self.burst_packet.frames:
                self.review_title.text=('Photo' if self.review_photo else 'Frame')+f' · {self.frame_index+1}/{len(self.burst_packet.frames)}'
        def update_view(self):
            target=self.photo_view if self.review_photo else self.cloud_view
            if target.parent is not self.viewport:
                # Remove only the previous image; never delete rotation arrows.
                for image in (self.cloud_view,self.photo_view):
                    if image.parent is self.viewport:self.viewport.remove_widget(image)
                self.viewport.add_widget(target,index=len(self.viewport.children))
            for arrow in self.rotation_buttons.values():
                if self.review_photo and arrow.parent is self.viewport:self.viewport.remove_widget(arrow)
                elif not self.review_photo and arrow.parent is None:self.viewport.add_widget(arrow)
            self.refresh_controls()
        def toggle_landmarks(self):
            r=self.cloud_view.renderer;r.show_landmarks=not r.show_landmarks;self.control_token=None;self.status_token=None;self.photo_token=None;self.update_panel();self.cloud_view.trigger();self.refresh_controls()
        def rotate_scan(self,yaw_delta=0,pitch_delta=0):
            if self.review_packet is None:return
            self.review_photo=False;self.cloud_view.gesture='rotate';self.update_view()
            r=self.cloud_view.renderer;r.yaw=(r.yaw+yaw_delta)%360
            r.pitch=float(np.clip(r.pitch+pitch_delta,-89,89))
            self.control_token=None;self.refresh_controls();self.cloud_view.trigger()
        def reset_view(self):
            self.cloud_view.renderer.reset(self.review_packet);self.remember_scan_view();self.cloud_view.trigger()
        def zoom(self,factor):
            r=self.cloud_view.renderer;r.zoom=float(np.clip(r.zoom*factor,.35,6));self.cloud_view.trigger()
        def open_distances(self):self.distance_dropdown.open(self.distance_menu)
        def other_asi_menu(self):
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(10))
            popup=Popup(title='Compare additional ASI formulas',content=content,size_hint=(.75,.65))
            for mode,title in [('old','Original ASI (requires both cheeks)'),('new','Horizontal + vertical ASI')]:
                content.add_widget(button(title,lambda m=mode:(popup.dismiss(),self.set_mode(m)),height=60))
            content.add_widget(label('Missing depth can prevent these formulas.\nSee the listed landmarks; use Scan result for the primary ASI.',13));popup.open()
        def change_result_scope(self):
            self.result_scope='frame' if self.result_scope=='scan' else 'scan'
            self.status_token=None;self.update_panel()
        def set_mode(self,mode):
            self.mode=mode
            for key,b in self.mode_buttons.items():b.tint=(.08,.42,.39,1) if key==mode else CARD
            self.cloud_view.renderer.mode=mode;self.cloud_view.renderer.selected_pair=self.selected_pair;self.cloud_view.renderer.include_depth=self.include_depth
            self.photo_token=None;self.rebuild_panel();self.update_panel();self.cloud_view.trigger()
        def rebuild_panel(self):
            key=(self.mode,bool(self.burst_packet and self.burst_packet.frames and not self.live_review),self.live_review)
            if key==self.panel_key:return
            self.panel_key=key
            self.status_token=None
            self.side_panel.clear_widgets();self.panel_labels={}
            def add(key,text,size=14,height=30,color=TEXT,bold=False):
                if key in ('delta',) and not self.debug_info:return None
                l=label(text,size,color,bold,size_hint_y=None,height=dp(height));self.side_panel.add_widget(l);self.panel_labels[key]=l;return l
            if self.mode=='summary':
                result=add('asi3','Loading scan result…',18,190,color=TEAL,bold=False);result.markup=True
                self.scope_button=button('Show frame ASI',self.change_result_scope,height=42);self.side_panel.add_widget(self.scope_button)
                self.side_panel.add_widget(button('Check eyelid opening / eyebrow movement',self.upper_checks,height=44,font_size=sp(12)))
                add('result_note','',12,56,color=MUTED)
                self.side_panel.add_widget(button('ASI graph across the recording',lambda:self.ensure_frames(self.show_asi_trace),height=44))
            if self.mode=='distance':
                add('hint','Tap two landmarks or select:',13,30,color=MUTED)
                add('distance','Unavailable',36,60,color=TEAL,bold=True)
                self.a_button=button(PRETTY[self.measure_pair[0]],lambda:self.choose_landmark(0),height=43);self.side_panel.add_widget(self.a_button)
                self.b_button=button(PRETTY[self.measure_pair[1]],lambda:self.choose_landmark(1),height=43);self.side_panel.add_widget(self.b_button)
                add('delta','',13,60,color=MUTED);add('quality','',12,70,color=MUTED)
                if self.burst_packet and self.burst_packet.frames and not self.live_review:self.side_panel.add_widget(button('Distance + nose Z over frames',self.selected_trace,height=44))
                self.side_panel.add_widget(button('How this is measured',lambda:notice('3D Euclidean distance','d = sqrt(dx^2 + dy^2 + dz^2)\n\nThe line connects two measured 3D landmarks in the same frame. XYZ and lengths are in mm. Landmark depth is a local robust same-frame depth median. Expanded or noisy patches are flagged under Depth quality. Raw geometry; displayed values use the selected moving window.'),height=40))
            elif self.mode in ('mouth','brow'):
                add('hint','Pronasale to cheilion' if self.mode=='mouth' else 'Trichion to superciliare',15,42,bold=True)
                add('lengths','',18,74,color=TEAL,bold=True);add('radial','',16,45)
                add('angles','',13,105);add('quality','',12,80,color=MUTED)
                if self.mode=='brow' and self.can_mark():self.side_panel.add_widget(button('Mark anatomical points',self.mark_landmarks,height=46))
                self.side_panel.add_widget(button('Definitions',self.anchor_formula,height=42))
            elif self.mode=='old':
                add('hint','Existing code · 3D extension',13,24,color=MUTED)
                add('score','Unavailable',38,62,color=TEAL,bold=True);add('regional','',13,32)
                add('quality','',12,100,color=MUTED)
                for i in range(4):add('pair'+str(i),'',13,34,color=MUTED)
                self.side_panel.add_widget(button('Formula + limitations',self.formula,height=40))
            elif self.mode=='new':
                add('hint','Proposed H + V metric',13,24,color=MUTED)
                add('score','Unavailable',38,62,color=TEAL,bold=True)
                add('quality','',12,110,color=MUTED)
                add('components','',14,70)
                self.pair_button=button('Illustrate: '+{'CHEILION':'Mouth corners','ALARE':'Nose wings','ZYGOMA':'Cheeks'}[self.selected_pair],self.choose_pair,height=40);self.side_panel.add_widget(self.pair_button)
                add('pair_values','',13,40,color=MUTED)
                self.depth_button=button('Include depth: OFF',self.toggle_depth,height=40);self.side_panel.add_widget(self.depth_button)
                self.side_panel.add_widget(button('Formula + limitations',self.formula,height=40))
            add('options_heading','DISPLAY OPTIONS',12,38,color=MUTED,bold=True)
            self.add_smoothing_controls(self.side_panel)
            for title,key in [('Key measurement landmarks','show_landmarks'),('Detailed face points','show_dense'),('Point cloud','show_cloud'),('Measurement lines','show_lines'),('Landmark labels','show_labels')]:
                row=BoxLayout(size_hint_y=None,height=dp(36));check=CheckBox(active=getattr(self.cloud_view.renderer,key),size_hint_x=None,width=dp(42))
                check.disabled=key=='show_lines' and self.mode!='distance'
                check.bind(active=lambda _,v,k=key:self.toggle_render_option(k,v));row.add_widget(check);row.add_widget(label(title,13));self.side_panel.add_widget(row)
            row=BoxLayout(size_hint_y=None,height=dp(36));check=CheckBox(active=self.debug_info,size_hint_x=None,width=dp(42))
            check.bind(active=lambda _,v:self.toggle_debug(v));row.add_widget(check);row.add_widget(label('Technical details',13));self.side_panel.add_widget(row)
            if self.debug_info:
                add('landmark_count','3D landmark count unavailable',12,49,color=MUTED)
                self.side_panel.add_widget(button('ASI formulas + coverage',self.three_asi_formula,height=43))
                self.side_panel.add_widget(button('Debug snapshot',self.debug_snapshot,height=43))
            if self.mode!='brow' and self.can_mark():self.side_panel.add_widget(button('Correct landmarks on photo',self.mark_landmarks,height=42))
            if self.debug_info:self.side_panel.add_widget(button('Depth quality / missing points',self.show_quality,height=42))
            self.side_scroll.scroll_y=1
            self.control_token=None;self.refresh_controls()
        def update_panel(self):
            packet=self.review_packet
            if packet is None:return
            m=packet.scores;old=m['old'];new=m['new']
            token=(packet.seq,packet.timestamp,packet.pose_mode,self.smooth_enabled,self.smooth_window,self.smooth_method,self.mode,self.measure_pair,self.selected_pair,self.include_depth,self.review_photo,self.cloud_view.renderer.show_landmarks,self.cloud_view.renderer.show_dense,self.result_scope)
            if token==self.status_token:return
            self.status_token=token
            def settext(k,text):
                if k in self.panel_labels:self.panel_labels[k].text=text
            asi=m['asi'];tally=landmark_tally(packet.landmarks,packet.landmark_pixels,packet.quality)
            upper=asi['coverage']['upper'];lower=asi['coverage']['lower'];overall=asi['coverage']['overall']
            summary=asi_display_summary(self.review_history(),self.smooth_window,self.smooth_enabled,self.smooth_method)
            values=summary['display']
            loading=self.burst_packet is not None and self.burst_packet.frames and not self.cached_frames() and not self.live_review
            if self.mode=='summary':
                frames=self.cached_frames() or [packet]
                whole=scan_asi_summary(frames)
                is_scan=self.result_scope=='scan' and not self.live_review
                values=whole['values'] if is_scan else summary['display']
                title='WHOLE-SCAN ASI' if is_scan else ('LIVE ASI' if self.live_review else 'FRAME ASI')
                averaging=(f"Scan mean: {whole['used']['overall']} valid frames" if is_scan else
                    (f"{summary['label']} ({summary['n_used']['overall']} valid)" if self.smooth_enabled else 'Smoothing off: current frame only'))
                settext('asi3', 'Loading scan result…' if loading else
                    f"[size=14]{title}[/size]\n[size=11]{averaging}[/size]\n[size=11]{expression_line(packet)}[/size]\n[size=48][b]{fmt(values['overall'],'%',1)}[/b][/size]\n"
                    f"[size=13]UPPER                         LOWER[/size]\n[size=32][b]{fmt(values['upper'],'%',1)}       {fmt(values['lower'],'%',1)}[/b][/size]")
                self.scope_button.text='Show frame ASI' if is_scan else 'Show whole-scan result'
                note=(f"Expression estimate follows the selected frame.\nAverage of {whole['used']['overall']} comparable frames (recorded: {whole['total']}).\nUpper: {whole['used']['upper']} frames. Lower: {whole['used']['lower']} frames."
                      if is_scan else summary['label']+f". Using {summary['n_used']['overall']} valid frames.")
                settext('result_note',note)
            missing=', '.join(tally['missing'][:2]) or 'none'
            settext('landmark_count',f"{packet.pose_mode} · 3D {tally['valid_3d']}/{tally['total']}  / 2D {tally['detected_2d']}/{tally['total']}\nMissing: {missing}")
            if self.mode=='distance':
                a,b=self.measure_pair;distance=self.displayed_value(lambda f:point_distance(f.landmarks,a,b));settext('distance',fmt(distance,' mm',2))
                if distance is not None:
                    delta=packet.landmarks[b]-packet.landmarks[a];settext('delta',f'Camera axes\n|dX| {abs(delta[0]):.1f}   |dY| {abs(delta[1]):.1f}\n|dZ| {abs(delta[2]):.1f} mm')
                else:settext('delta','Select two different valid landmarks.')
                self.a_button.text='A: '+PRETTY[a];self.b_button.text='B: '+PRETTY[b]
                self.a_button.tint=(.55,.27,.08,1);self.b_button.tint=(.40,.20,.52,1)
                settext('quality',f'{len(packet.landmarks)} measured landmarks\nDrag rotates · dots select A/B.\n'+self.pair_summary())
            elif self.mode in ('mouth','brow'):
                anchored=m['mouth'] if self.mode=='mouth' else (m['brow'] if 'TRICHION' in packet.landmarks else m['forehead_proxy'])
                settext('hint','Pronasale to cheilion' if self.mode=='mouth' else ('Trichion to superciliare' if 'TRICHION' in packet.landmarks else 'Forehead PROXY to brow proxies'))
                settext('lengths',f"Left  {fmt(anchored['left_mm'],' mm',2)}\nRight {fmt(anchored['right_mm'],' mm',2)}")
                settext('radial',f"Radial asymmetry {fmt(anchored['radial_asi_pct'],'%')}")
                settext('angles',f"Angle to face-down axis\nL {fmt(anchored['left_inclination_deg'],' deg')} / R {fmt(anchored['right_inclination_deg'],' deg')}\nDifference {fmt(anchored['inclination_difference_deg'],' deg')}\nOpening angle {fmt(anchored['opening_angle_deg'],' deg')}\nVertical gap {fmt(anchored['vertical_mm'],' mm')}")
                settext('quality', 'Lengths use measured XYZ.\nAngles need chin + inner eyes.' if self.mode=='mouth' else 'Automatic brow points are proxies.\nMark actual hairline and brows\non the frozen photo for anatomy.\nNo true trichion is auto detected.')
            elif self.mode=='old':
                settext('score',fmt(self.displayed_value(lambda f:None if f.scores['old'] is None else f.scores['old']['AsI_overall']*100),' %',2))
                settext('regional',f"Upper {fmt(self.displayed_value(lambda f:None if f.scores['old'] is None else f.scores['old']['AsI_upper']*100),'%')}   Lower {fmt(self.displayed_value(lambda f:None if f.scores['old'] is None else f.scores['old']['AsI_lower']*100),'%')}")
                names=['Eyes','Nose','Mouth','Cheeks']
                for i,row in enumerate(m['rows']):settext('pair'+str(i),f"{names[i]}  L {fmt(row['old_left_mm'])} / R {fmt(row['old_right_mm'])} mm")
                settext('quality', 'Needs depth: '+', '.join(PRETTY.get(k,k) for k in m['old_missing']) if m['old_missing'] else 'Distances to historical midline plane.\nInherited calculation unchanged.')
            elif self.mode=='new':
                full=new['xyz_pct'] if self.include_depth else new['hv_pct'];core=new['core_xyz_pct'] if self.include_depth else new['core_hv_pct']
                key=('xyz_pct' if self.include_depth else 'hv_pct') if full is not None else ('core_xyz_pct' if self.include_depth else 'core_hv_pct')
                score=self.displayed_value(lambda f:f.scores['new'][key]);settext('score',fmt(score,' %',2))
                settext('hint',('Full 3 pairs' if full is not None else 'Core nose + mouth (2 pairs)')+(' H+V+D' if self.include_depth else ' H+V'))
                settext('components',f"Horizontal  {fmt(new['horizontal_mm'],' mm')}\nVertical       {fmt(new['vertical_mm'],' mm')}\nDepth           {fmt(new['depth_mm'],' mm')}")
                row=next((r for r in m['rows'][1:] if r['left'].startswith(self.selected_pair+'_')),None)
                settext('pair_values','' if row is None else f"Selected pair: H {fmt(row['horizontal_mm'])} / V {fmt(row['vertical_mm'])}\nDepth {fmt(row['depth_mm'])} mm")
                direction=''
                if row and row['signed_vertical_mm'] is not None:
                    value=row['signed_vertical_mm']
                    direction='Right lower' if value>0.5 else 'Right higher' if value<-.5 else 'Similar height'
                settext('quality',f"{direction} · {new['pairs_valid']}/3 valid\nMouth H+V {fmt(new['mouth_hv_pct'],'%')}\nFull {fmt(full,'%')} · Core {fmt(core,'%')}")
                if score is None:settext('quality','Needs depth: '+', '.join(PRETTY.get(k,k) for k in m['new_missing'])+'\nReposition the face; missing coordinates are not filled in.')
                self.depth_button.text='Include depth: ON' if self.include_depth else 'Include depth: OFF'
            self.cloud_view.renderer.measure_pair=self.measure_pair if self.mode=='distance' else ()
            if self.review_photo and token!=self.photo_token:
                self.photo_token=token;self.photo_view.show_review(packet,self.mode,self.selected_pair,self.include_depth,self.measure_pair if self.mode=='distance' else None,self.cloud_view.renderer.show_landmarks,f'frame {self.frame_index+1}' if self.burst_packet and self.burst_packet.frames else 'this capture',self.cloud_view.renderer.show_dense)
        def choose_landmark(self,side):
            content=BoxLayout(orientation='vertical',padding=dp(12),spacing=dp(10));search=TextInput(hint_text='Search landmark',multiline=False,size_hint_y=None,height=dp(44),font_size=sp(16))
            content.add_widget(search);scroll=ArrowScroll(do_scroll_x=False);items=BoxLayout(orientation='vertical',spacing=dp(6),size_hint_y=None);items.bind(minimum_height=items.setter('height'));scroll.add_widget(items);content.add_widget(scroll)
            popup=Popup(title='Select point A' if side==0 else 'Select point B',content=content,size_hint=(.75,.9))
            def rebuild(*_):
                items.clear_widgets();query=search.text.casefold()
                for name in list(LM)+['TRICHION']:
                    # Missing recorded landmarks remain selectable and report unavailable.
                    if query not in PRETTY[name].casefold() and query not in name.casefold():continue
                    valid=self.review_packet and name in self.review_packet.landmarks
                    b=button(PRETTY[name]+('' if valid else '  · no depth'),lambda n=name:self.select_and_close(n,side,popup),height=45)
                    items.add_widget(b)
            content.add_widget(button('Cancel',popup.dismiss,height=42));search.bind(text=rebuild);rebuild();popup.open()
        def select_and_close(self,name,side,popup):self.picked(name,side);popup.dismiss()
        def picked(self,name,side):
            pair=list(self.measure_pair);pair[side]=name;self.measure_pair=tuple(pair);self.cloud_view.renderer.measure_pair=self.measure_pair
            if self.mode!='distance':self.set_mode('distance')
            self.status_token=None;self.photo_token=None;self.update_panel();self.cloud_view.trigger()
        def choose_pair(self):
            content=BoxLayout(orientation='vertical',padding=dp(16),spacing=dp(10));popup=Popup(title='Which pair should we illustrate?',content=content,size_hint=(.7,.6))
            for name,title in [('ALARE','Nose wings'),('CHEILION','Mouth corners'),('ZYGOMA','Cheeks')]:
                def choose(n=name,t=title):
                    self.selected_pair=n;self.pair_button.text='Illustrate: '+t;self.cloud_view.renderer.selected_pair=n;self.status_token=None;self.photo_token=None;self.update_panel();self.cloud_view.trigger();popup.dismiss()
                content.add_widget(button(title,choose,height=50))
            content.add_widget(button('Cancel',popup.dismiss,height=42));popup.open()
        def toggle_depth(self):
            self.include_depth=not self.include_depth;self.cloud_view.renderer.include_depth=self.include_depth;self.status_token=None;self.photo_token=None;self.update_panel();self.cloud_view.trigger()
        def formula(self):
            if self.mode=='old':text=('OLD 3D\nPair = abs(dR - dL) / (dR + dL + 1e-6)\nASI % = 100 x mean of 4 pairs.\n\ndL and dR are perpendicular 3D distances to the estimated midline plane. The inherited code is preserved. Its clinical study protocol is not supplied.\n\nMidline or landmark drift changes the estimate. This is not a diagnosis.')
            else:text=('PROPOSED H + V\nReflect left point in the facial midline plane.\nResidual = right - reflected left.\nFor each pair: e = sqrt(H^2 + V^2).\nASI % = 100 x mean(e) / inner-eye distance.\n\nOptional depth: sqrt(H^2 + V^2 + D^2).\nFull: nose wings, mouth, cheeks (3 pairs).\nCore: nose wings + mouth (2 required pairs).\nMouth-only is separately labeled. No variable pair averaging.\n\nFace axes use glabella, nose, chin and inner eyes. Reference drift biases the result. This proposed score has no validated clinical cutoff and may exceed 100%.')
            notice('Calculation',text)
        def choose_export(self,folder=None):
            if self.exporting or self.saving:return
            root=self.burst_packet or self.review_packet
            folder=Path(folder) if folder else (Path(root.path) if root and root.path else None)
            if folder is None or not folder.is_dir():notice('Save first','Wait for capture saving to finish.');return
            destination=configured_scans_dir()
            if destination is None:
                self.export_folder_popup(folder);return
            self.begin_export(folder,destination)
        def export_folder_popup(self,folder=None,on_selected=None):
            from kivy.uix.filechooser import FileChooserListView
            content=BoxLayout(orientation='vertical',padding=dp(10),spacing=dp(6))
            popup=Popup(title='Choose CSV results folder',content=content,size_hint=(.98,.98))
            content.add_widget(label('Select your existing Dropbox scans/results folder. It will be remembered.',12,size_hint_y=None,height=dp(32)))
            field=TextInput(text=str(configured_scans_dir() or Path.home()),multiline=False,size_hint_y=None,height=dp(42))
            content.add_widget(field)
            chooser=FileChooserListView(path=str(Path.home()),dirselect=True,filters=[lambda directory,name:(Path(directory)/name).is_dir()])
            content.add_widget(chooser)
            status=label('',12,color=WARN,size_hint_y=None,height=dp(30));content.add_widget(status)
            def navigate(path):
                try:
                    p=Path(path).expanduser()
                    if not p.is_absolute() or not p.is_dir():raise ValueError('Folder not found')
                    chooser.path=str(p);chooser.selection=[];field.text=str(p);status.text=''
                except (ValueError,OSError) as exc:status.text=str(exc)
            chooser.bind(path=lambda _,v:setattr(field,'text',v))
            chooser.bind(selection=lambda _,v:setattr(field,'text',v[0]) if v else None)
            field.bind(on_text_validate=lambda *_:navigate(field.text))
            shortcuts=BoxLayout(size_hint_y=None,height=dp(38),spacing=dp(5))
            shortcuts.add_widget(button('Home',lambda:navigate(Path.home()),height=38))
            shortcuts.add_widget(button('Open path',lambda:navigate(field.text),height=38))
            choices=export_folder_candidates()
            if choices:
                pick=Spinner(text='Found results folders',values=[str(p) for p in choices],font_size=sp(11))
                pick.bind(text=lambda _,v:navigate(v) if v in pick.values else None);shortcuts.add_widget(pick)
            content.add_widget(shortcuts)
            def select(destination=None,local=False):
                try:
                    target=Path(destination or field.text.strip()).expanduser()
                    if local:target.mkdir(parents=True,exist_ok=True)
                    target=remember_export_folder(target,folder)
                except (OSError,ValueError) as exc:status.text=str(exc);return
                popup.dismiss()
                if on_selected:on_selected(target)
                elif folder:self.begin_export(folder,target)
            actions=BoxLayout(size_hint_y=None,height=dp(46),spacing=dp(6))
            actions.add_widget(button('Cancel',popup.dismiss,height=46))
            actions.add_widget(button('Local results',lambda:select(DATA_ROOT.parent/'exported_scans',True),height=46))
            actions.add_widget(button('Use folder / Export' if folder else 'Use folder',select,height=46,tint=TEAL))
            content.add_widget(actions);popup.open()
        def begin_export(self,folder,destination):
            if self.exporting:return
            self.last_export=None;self.exporting=True;self.export_progress='Preparing CSV export…';self.mark_name=None;self.photo_view.on_mark=None;self.stop_playback();self.control_token=None;self.refresh_controls()
            content=BoxLayout(orientation='vertical',padding=dp(14),spacing=dp(10));status=label('Preparing CSV export…',15);content.add_widget(status)
            popup=Popup(title='Exporting results',content=content,size_hint=(.88,.6),auto_dismiss=False);popup.open()
            def update(done,total):
                self.export_progress=f'Writing frame {done}/{total}';Clock.schedule_once(lambda *_:setattr(status,'text',self.export_progress),0)
            def worker():
                try:
                    path,count=export_matlab_csv(folder,destination,update,self.smooth_window if self.smooth_enabled else 1,self.smooth_method)
                    def success(*_):
                        self.last_export=path;self.exporting=False;self.control_token=None;self.refresh_controls();popup.dismiss()
                        notice('CSV ready for sync',f'{count} frames in one CSV:\n{path}\n\nYour Dropbox sync service must finish uploading before it appears on the computer.')
                    Clock.schedule_once(success,0)
                except Exception as exc:
                    message=str(exc)
                    def failure(*_):
                        self.exporting=False;self.control_token=None;self.refresh_controls();popup.dismiss();notice('Export failed',message+'\n\nThe original capture is unchanged. Reconnect the destination and try again.')
                    Clock.schedule_once(failure,0)
            thread=threading.Thread(target=worker,daemon=False);self.export_threads.append(thread);thread.start()
        def refresh_library(self):
            self.library_items.clear_widgets();folders=sorted([p for p in DATA_ROOT.glob('*/metadata.csv') if not p.parent.name.startswith('.')],reverse=True) if DATA_ROOT.exists() else []
            if not folders:
                self.library_items.add_widget(label('No saved captures yet.\nTake a single capture to begin.',19,color=MUTED,size_hint_y=None,height=dp(110)))
            for meta_path in folders:
                try:
                    with meta_path.open() as f:meta={r['key']:r['value'] for r in csv.DictReader(f)}
                    title=f"{meta.get('patient_id','Capture')}  ·  {datetime.fromtimestamp(float(meta['timestamp_s'])).strftime('%b %d  %H:%M:%S')}"
                    detail=f"{meta.get('pose_mode','Unspecified')} · {meta.get('capture_frames',1)} frame(s) · CSV saved"+(' · DEMO' if meta.get('demo')=='1' else '')
                    row=BoxLayout(size_hint_y=None,height=dp(78),spacing=dp(8))
                    row.add_widget(button(title+'\n'+detail,lambda p=meta_path.parent:self.open_saved(p),height=78))
                    row.add_widget(button('Export\nCSV',lambda p=meta_path.parent:self.choose_export(p),height=78,size_hint_x=None,width=dp(95),tint=TEAL))
                    self.library_items.add_widget(row)
                except Exception:traceback.print_exc()
        def open_saved(self,folder):
            if folder in self.session_packets:
                self.live_review=False;self.live_button.text='Live 3D';self.review_photo=False;self.load_review(self.session_packets[folder]);self.go('review');return
            if self.review_packet and self.review_packet.path==folder:
                self.live_review=False;self.live_button.text='Live 3D';self.load_review(self.review_packet);self.go('review');return
            self.home_status.text='Loading capture…'
            def load():
                try:
                    packet=load_packet(folder);Clock.schedule_once(lambda *_:self.saved_loaded(packet),0)
                except Exception as exc:
                    message=str(exc);Clock.schedule_once(lambda *_:notice('Could not open capture',message),0)
            threading.Thread(target=load,daemon=True).start()
        def saved_loaded(self,packet):
            self.live_review=False;self.live_button.text='Live 3D';self.review_photo=False;self.load_review(packet);self.go('review')
        def tick(self,dt):
            self.home_status.text=self.engine.error or self.engine.state
            if self.engine.error:self.home_status.color=WARN
            if self.manager.current not in ('capture','review'):return
            packet=self.engine.snapshot();fresh=packet is not None and packet_age(packet)<STALE_SECONDS
            if self.manager.current=='capture':
                allowed=fresh and len(packet.points)>=100 and not self.saving
                self.capture_button.disabled=not allowed;self.capture_button.opacity=1 if allowed else 0
                if self.saving:
                    self.guide_text.text=self.capture_progress
                self.capture_button.text=f'Capture {args.frames}F' if fresh and packet.guide.get('ready') else f'Capture {args.frames}F\ncheck guidance'
                self.capture_button.tint=TEAL if fresh and packet.guide.get('ready') else (.55,.36,.12,1)
                if not fresh:
                    self.distance_title.text='No fresh depth data';self.distance_title.color=WARN
                    self.guide_text.text=self.engine.error or 'Waiting for OAK camera…';self.capture_numbers.text='ASI unavailable · 3D landmarks unavailable';self.live_values.text='ASI unavailable · no fresh frame';return
                if packet.seq==self.last_seq:return
                self.last_seq=packet.seq
                if self.capture_cloud:self.capture_cloud_view.set_packet(packet)
                elif packet.rgb is not None:self.feed.show(dense_overlay(packet.rgb,packet) if self.capture_dense else (landmark_overlay(packet.rgb,packet) if self.capture_points else packet.rgb))
                if self.capturing:self.guide_text.text=self.capture_progress
                distance=packet.guide['distance_mm'];distance_ok=distance is not None and TARGET_MM-TOLERANCE_MM<=distance<=TARGET_MM+TOLERANCE_MM
                self.distance_title.text=(f'{distance/10:.1f} cm'+(' · distance OK' if distance_ok else ' · outside range')) if distance is not None else 'Nose depth unavailable'
                self.distance_title.color=TEAL if distance_ok else WARN;self.guide_text.text=self.capture_progress if self.saving else packet.guide['message']
                asi=packet.scores['asi'];tally=landmark_tally(packet.landmarks,packet.landmark_pixels,packet.quality)
                up=asi['coverage']['upper'];lo=asi['coverage']['lower'];whole=asi['coverage']['overall']
                missing=', '.join(tally['missing'][:3]) or 'none'
                if len(tally['missing'])>3:missing+=f' (+{len(tally["missing"])-3} more)'
                with self.engine.lock:live_frames=list(self.engine.metric_history)
                live_frames=[f for f in live_frames if f.seq<=packet.seq and f.timestamp<=packet.timestamp] or [packet]
                summary=asi_display_summary(live_frames,self.smooth_window,self.smooth_enabled,self.smooth_method)
                live_asi=summary['display']
                self.live_values.height=dp(92) if self.live_asi_enabled else 0
                self.live_values.opacity=1 if self.live_asi_enabled else 0
                self.live_values.text=(f"[size=13]LIVE ASI | {summary['label']} | {summary['n_used']['overall']} valid frames[/size]\n"
                    +f"[size=28][b]Overall {fmt(live_asi['overall'],'%')}[/b]   Upper {fmt(live_asi['upper'],'%')}   Lower {fmt(live_asi['lower'],'%')}[/size]\n[size=12]{expression_line(packet)}[/size]")
                self.capture_numbers.text=(
                    f"{self.pose_mode.upper()} · UPPER {fmt(asi['upper'],'%',1)} ({up[0]}/{up[1]})  · LOWER {fmt(asi['lower'],'%',1)} ({lo[0]}/{lo[1]})\n"
                    f"OVERALL 3D ASI {fmt(asi['overall'],'%',1)} ({whole[0]}/{whole[1]} pairs) · experimental\n"
                    f"3D LANDMARKS {tally['valid_3d']}/{tally['total']}  · 2D detected {tally['detected_2d']}/{tally['total']}  · depth good {tally['good_depth']}\n"
                    f"Missing: {missing} · {self.engine.fps:.1f} fps" + (' · DEMO' if packet.demo else ''))
            elif self.live_review:
                if fresh and packet.seq!=self.last_seq:
                    self.last_seq=packet.seq;self.review_packet=packet;self.cloud_view.set_packet(packet);self.review_title.text='LIVE · '+f'{self.engine.fps:.1f} fps'+(' · DEMO' if packet.demo else '');self.update_panel()
                elif not fresh:
                    self.review_title.text='Live stream paused / unavailable'
                    # Do not present the previous frame's numbers as live measurements.
                    for key in ['asi3','score','distance','components','delta','regional','pair_values','pair0','pair1','pair2','pair3']:
                        if key in self.panel_labels:self.panel_labels[key].text='Unavailable · no fresh frame'
                    self.status_token=None
            else:self.update_panel()
            self.refresh_controls()
        def screenshots(self,dt):
            # Developer-only screenshot harness, always explicitly synthetic.
            args.screenshot_dir.mkdir(parents=True,exist_ok=True)
            self.manager.current='home'
            def shoot(name,after):
                self.root.export_to_png(str(args.screenshot_dir/name));after()
            def cap():self.open_capture();Clock.schedule_once(lambda *_:shoot('capture.png',review),3.0)
            def review():
                packet=self.engine.snapshot() or demo_packet();self.engine.active.clear();self.live_review=False;self.load_review(packet);self.manager.current='review';self.set_mode('new')
                Clock.schedule_once(lambda *_:shoot('review_new.png',old),.5)
            def old():self.set_mode('old');Clock.schedule_once(lambda *_:shoot('review_old.png',mouth),.5)
            def mouth():self.set_mode('mouth');Clock.schedule_once(lambda *_:shoot('review_mouth.png',brow),.5)
            def brow():self.set_mode('brow');Clock.schedule_once(lambda *_:shoot('review_brow.png',dist),.5)
            def dist():self.set_mode('distance');Clock.schedule_once(lambda *_:shoot('review_distance.png',self.ui_smoke_test),.5)
            Clock.schedule_once(lambda *_:shoot('home.png',cap),.3)
        def ui_smoke_test(self):
            assert self.live_button.parent is None
            assert self.cloud_view.renderer.color_mode=='solid'
            assert not self.cloud_view.renderer.show_lines
            self.set_mode('distance');self.measure_pair=('GLABELLA','GLABELLA');self.status_token=None;self.update_panel()
            assert '0.00' in self.panel_labels['distance'].text
            self.set_smoothing(window=30,method='median');self.toggle_debug(True)
            assert 'landmark_count' in self.panel_labels
            self.toggle_debug(False);assert 'landmark_count' not in self.panel_labels
            self.toggle_render_option('show_dense',True);assert self.cloud_view.renderer.show_dense
            self.toggle_render_option('show_dense',False)
            self.go('capture');self.set_capture_dense(True);assert self.capture_cloud_view.renderer.show_dense
            checks=self.upper_checks();Clock.schedule_once(lambda *_:checks.dismiss(),.25)
            self.set_capture_dense(False);self.toggle_capture_view()
            assert self.capture_cloud_view.parent is self.capture_stage
            self.toggle_capture_view();assert self.feed.parent is self.capture_stage
            self.select_pose('Smiling');Clock.schedule_once(lambda *_:self.take_capture(),.6)
            self.smoke_deadline=time.monotonic()+120
            Clock.schedule_once(self.verify_new_capture,1.0)
        def verify_new_capture(self,*_):
            if self.saving:
                assert time.monotonic()<self.smoke_deadline
                Clock.schedule_once(self.verify_new_capture,.2);return
            assert self.burst_packet and len(self.burst_packet.frames)==args.frames
            assert self.manager.current=='review'
            self.set_mode('summary');self.show_frame(len(self.burst_packet.frames)-1)
            assert 'WHOLE-SCAN' in self.panel_labels['asi3'].text
            self.change_result_scope()
            assert 'FRAME ASI' in self.panel_labels['asi3'].text
            self.set_smoothing(window=1,method='mean');assert 'N=1' in self.panel_labels['result_note'].text
            self.set_smoothing(window=20,method='mean');assert 'N=20' in self.panel_labels['result_note'].text
            assert 'N=20' in self.panel_labels['asi3'].text
            self.show_frame(0);assert 'Using 1 valid frames' in self.panel_labels['result_note'].text
            self.show_frame(len(self.burst_packet.frames)-1)
            self.set_smoothing(window=60,method='mean');assert self.smooth_window==60
            self.change_result_scope();assert 'WHOLE-SCAN' in self.panel_labels['asi3'].text
            assert 'Scan mean:' in self.panel_labels['asi3'].text
            self.cloud_view.renderer.show_lines=True;self.cloud_view.render()
            Clock.schedule_once(lambda *_:self.root.export_to_png(str(args.screenshot_dir/'review_smoothed.png')),.1)
            destination=args.screenshot_dir/'exports';destination.mkdir(exist_ok=True)
            self.begin_export(self.burst_packet.path,destination)
            Clock.schedule_once(self.verify_new_export,.2)
        def verify_new_export(self,*_):
            if self.exporting:
                assert time.monotonic()<self.smoke_deadline
                Clock.schedule_once(self.verify_new_export,.2);return
            assert self.last_export.suffix=='.csv'
            with self.last_export.open(newline='') as f:
                rows=[r for r in csv.DictReader(f) if r['record_type']=='landmark']
            assert len(rows)==args.frames*21
            (args.screenshot_dir/'ui_test_result.txt').write_text('PASS: capture camera/cloud toggle, hidden debug, selectable smoothing, same-point measurement, complete capture, review, single-CSV export.\n')
            self.stop()

        def on_stop(self):
            for widget in self.root.walk():
                if isinstance(widget,ArrowScroll):widget.stop_scroll()
            self.stop_playback();self.engine.stop();self.engine.join(timeout=2)
    FaceApp().run()

if __name__=='__main__':main()
