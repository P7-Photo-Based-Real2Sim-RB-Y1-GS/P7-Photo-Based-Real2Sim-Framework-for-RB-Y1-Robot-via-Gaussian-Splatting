#!/usr/bin/env python3
"""Small UDP test sender for Isaac bridge without webcam/ROS.
It sends sinusoidal left/right wrist motion so you can verify robot movement first.
"""
from __future__ import annotations
import argparse, json, math, socket, time


def lm(x,y,z):
    return {"x":x,"y":y,"z":z,"visibility":1.0,"presence":1.0,"valid":True}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=50555); ap.add_argument('--hz',type=float,default=30.0)
    args=ap.parse_args(); sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); t0=time.time(); dt=1.0/args.hz
    print(f"[FAKE] sending to {args.host}:{args.port}")
    while True:
        t=time.time()-t0
        # human basis points: left/right shoulders/hips and moving wrists
        landmarks={
            "left_shoulder": lm(-0.22,0.0,1.45), "right_shoulder": lm(0.22,0.0,1.45),
            "left_hip": lm(-0.16,0.0,0.95), "right_hip": lm(0.16,0.0,0.95),
            "left_elbow": lm(-0.45,0.12,1.25), "right_elbow": lm(0.45,0.12,1.25),
            "left_wrist": lm(-0.60,0.18+0.12*math.sin(t),1.10+0.12*math.sin(0.7*t)),
            "right_wrist": lm(0.60,0.18+0.12*math.sin(t+math.pi),1.10+0.12*math.sin(0.7*t+math.pi)),
        }
        pkt={"stamp_sec":time.time(),"frame_id":"fake","coordinate_type":"stitched_world","tracking_ok":True,"mean_visibility":1.0,"landmarks":landmarks}
        sock.sendto(json.dumps(pkt,separators=(',',':')).encode(),(args.host,args.port))
        time.sleep(dt)
if __name__=='__main__': main()
