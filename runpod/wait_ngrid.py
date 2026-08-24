"""Exit as soon as the controlled re-grow finishes (or fails). One notification."""
import json, os, subprocess, sys, time
HERE=r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
KEY=os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
p=json.load(open(os.path.join(HERE,"grow_pod_ngrid.json")))
def ssh(cmd,t=60):
    return subprocess.run(["ssh","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=NUL",
        "-o","ConnectTimeout=20","-i",KEY,"-p",str(p["port"]),f"root@{p['ip']}",cmd],
        capture_output=True,text=True,timeout=t)
deadline=time.time()+40*60
while time.time()<deadline:
    r=ssh("cat /work/RESULT_LIST.txt 2>/dev/null | wc -l; grep -c 'ALL DONE' /work/ng_0.log 2>/dev/null; "
          "ls /work/paths_ng 2>/dev/null | wc -l; grep -l 'SEED_.*_EXIT' /work/ng_*.log 2>/dev/null | wc -l")
    o=(r.stdout or "").split()
    done=len([x for x in o if x.isdigit()])>=4 and int(o[3])>=8
    if done:
        print("GROW COMPLETE"); print(ssh("cat /work/NGRID_PROOF.txt; echo ---; ls /work/paths_ng").stdout)
        sys.exit(0)
    time.sleep(45)
print("TIMEOUT after 40 min"); sys.exit(1)
