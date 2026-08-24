"""Run the grow with the SSH channel HELD OPEN from this side.
`nohup ... & disown` inside a one-shot ssh exec dies when the channel closes on this
image; the process never even created /work. Holding the connection is reliable."""
import json, os, subprocess, sys, time
HERE=r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
KEY=os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
p=json.load(open(os.path.join(HERE,"grow_pod_ngrid.json")))
seeds=json.load(open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\seeds_0813.json"))
arg=" ".join(f"{s['x']},{s['y']},{s['z']}" for s in seeds)
cmd=f"sed -i 's/\r$//' /grow_0813_ngrid.sh && bash /grow_0813_ngrid.sh '{arg}'"
print(f"[{time.strftime('%H:%M:%S')}] starting held-open grow", flush=True)
r=subprocess.run(["ssh","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=NUL",
    "-o","ConnectTimeout=25","-o","ServerAliveInterval=30","-o","ServerAliveCountMax=240",
    "-i",KEY,"-p",str(p["port"]),f"root@{p['ip']}",cmd],
    capture_output=True,text=True,timeout=3600)
print(f"[{time.strftime('%H:%M:%S')}] rc={r.returncode}", flush=True)
tail=(r.stdout+r.stderr).strip().split("\n")
print("\n".join(tail[-25:]), flush=True)
