#!/usr/bin/env python3
"""Author camera trajectories for Matrix-Game 3.5.

Convention (read off the bundled samples, all of which agree):
  extrinsics_c2w : (N,4,4) float32 camera-to-world. Frame 0 is IDENTITY --
                   camera starts at the origin. Axes: +X right, +Y up, +Z forward.
  intrinsics     : (N,4) float32 [fx, fy, cx, cy] in pixels of the anchor image.
                   Constant across frames in every sample. cx,cy = 639.05,359.05
                   for a 1280x720 anchor. fx sets FOV:
                     fx~647  -> ~89 deg (wide)      fx~794 -> ~78 deg
                     fx~1078 -> ~61 deg (tele)
  Scale: samples move 0.8-17 units over 961 frames; per-frame step 0.002-0.03.
         Treat ~1 unit as "a room", and keep steps small or the model tears.

Usage:
  python make_camera.py out.npz dolly    --dist 8
  python make_camera.py out.npz truck    --dist 4
  python make_camera.py out.npz pedestal --dist 2
  python make_camera.py out.npz pan      --deg 60
  python make_camera.py out.npz tilt     --deg -25
  python make_camera.py out.npz orbit    --radius 6 --deg 90
  python make_camera.py out.npz crane    --dist 5 --rise 3 --deg -20
  ... plus --frames 961 --fx 647
"""
import argparse, numpy as np

def Rx(t): c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(t): c,s=np.cos(t),np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def build(move, n, dist, deg, radius, rise):
    E=np.tile(np.eye(4,dtype=np.float32),(n,1,1))
    t=np.linspace(0,1,n)
    ease=t*t*(3-2*t)                      # smoothstep: no jerk at either end
    rad=np.deg2rad(deg)
    for i in range(n):
        R=np.eye(3); p=np.zeros(3)
        if   move=="dolly":    p=np.array([0,0,dist*ease[i]])
        elif move=="truck":    p=np.array([dist*ease[i],0,0])
        elif move=="pedestal": p=np.array([0,dist*ease[i],0])
        elif move=="pan":      R=Ry(rad*ease[i])
        elif move=="tilt":     R=Rx(rad*ease[i])
        elif move=="orbit":
            a=rad*ease[i]
            p=np.array([radius*np.sin(a),0,radius*(1-np.cos(a))])
            R=Ry(a)                        # keep the subject centred
        elif move=="crane":
            p=np.array([0,rise*ease[i],dist*ease[i]]); R=Rx(rad*ease[i])
        else: raise SystemExit(f"unknown move: {move}")
        E[i,:3,:3]=R; E[i,:3,3]=p
    return E.astype(np.float32)

ap=argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("move")
ap.add_argument("--frames",type=int,default=961)
ap.add_argument("--dist",type=float,default=8.0)
ap.add_argument("--deg",type=float,default=60.0)
ap.add_argument("--radius",type=float,default=6.0)
ap.add_argument("--rise",type=float,default=3.0)
ap.add_argument("--fx",type=float,default=647.0)
ap.add_argument("--width",type=int,default=1280); ap.add_argument("--height",type=int,default=720)
a=ap.parse_args()

E=build(a.move,a.frames,a.dist,a.deg,a.radius,a.rise)
K=np.tile(np.array([a.fx,a.fx,(a.width-1)/2+0.05*0,(a.height-1)/2],dtype=np.float32),(a.frames,1))
K[:,2]=639.0476 if a.width==1280 else (a.width-1)/2
K[:,3]=359.0476 if a.height==720 else (a.height-1)/2
np.savez(a.out, extrinsics_c2w=E, intrinsics=K.astype(np.float32))
step=np.median(np.linalg.norm(np.diff(E[:,:3,3],axis=0),axis=1))
fov=2*np.degrees(np.arctan((a.width/2)/a.fx))
print(f"wrote {a.out}: {a.frames} frames, move={a.move}, "
      f"path={np.linalg.norm(np.diff(E[:,:3,3],axis=0),axis=1).sum():.2f}u, "
      f"median step={step:.4f}, fov={fov:.1f}deg")
