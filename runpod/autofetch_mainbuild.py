"""Poll a grow pod, fetch its output, then TERMINATE IT — without needing anyone watching.

Written after two incidents where a pod idled for hours because the watcher only
printed a warning: pod1 idled ~8 h (~$5.90) and grow-0813-ngrid idled ~11 h (~$3.60).
A watcher that reports but cannot act is not a safeguard. This one terminates.

It stops the pod on ANY exit path — success, failure, or deadline — and only after the
output has been pulled locally and verified non-empty.
"""
import json
import os
import subprocess
import sys
import time

HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
DST = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\pherc0813_mainbuild"
KEY = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
POLL_S = 60
DEADLINE_MIN = 45

sys.path.insert(0, HERE)
import rp  # noqa: E402


def main():
    pod = json.load(open(os.path.join(HERE, "grow_pod_ngrid.json")))
    pid, ip, port = pod["id"], pod["ip"], pod["port"]
    os.makedirs(DST, exist_ok=True)

    def ssh(cmd, t=120):
        try:
            r = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                                "-o", "ConnectTimeout=20", "-i", KEY, "-p", str(port), f"root@{ip}", cmd],
                               capture_output=True, text=True, timeout=t)
            return (r.stdout + r.stderr).strip()
        except Exception as e:
            return f"__SSH_FAIL__ {type(e).__name__}"

    def terminate(reason):
        try:
            rp.terminate(pid)
            print(f"[{time.strftime('%H:%M:%S')}] TERMINATED {pid} — {reason}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] TERMINATE FAILED {pid}: {e} — DO THIS BY HAND", flush=True)

    deadline = time.time() + DEADLINE_MIN * 60
    fetched = False
    try:
        while time.time() < deadline:
            n_done = ssh("grep -h 'M_.*_EXIT' /work/M_*.log 2>/dev/null | wc -l")
            n_out = ssh("ls /work/paths_M 2>/dev/null | wc -l")
            print(f"[{time.strftime('%H:%M:%S')}] exits={n_done} patches={n_out}", flush=True)
            if n_done.isdigit() and int(n_done) >= 8:
                ssh("cd /work && tar czf /work/ng2.tgz paths_M M_*.log seed_M.json 2>/dev/null; ls -la /work/ng2.tgz")
                r = subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                                    "-i", KEY, "-P", str(port), f"root@{ip}:/work/ng2.tgz",
                                    os.path.join(DST, "ng2.tgz")], capture_output=True, text=True, timeout=600)
                p = os.path.join(DST, "ng2.tgz")
                if r.returncode == 0 and os.path.exists(p) and os.path.getsize(p) > 10000:
                    subprocess.run(["tar", "xzf", "ng2.tgz"], cwd=DST, capture_output=True)
                    fetched = True
                    print(f"[{time.strftime('%H:%M:%S')}] FETCHED {os.path.getsize(p)} bytes -> {DST}", flush=True)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] FETCH FAILED rc={r.returncode} — NOT terminating yet", flush=True)
                    time.sleep(POLL_S)
                    continue
                break
            time.sleep(POLL_S)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] deadline hit with exits={n_done}", flush=True)
            ssh("cd /work && tar czf /work/ng2.tgz paths_M M_*.log 2>/dev/null")
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                            "-i", KEY, "-P", str(port), f"root@{ip}:/work/ng2.tgz",
                            os.path.join(DST, "ng2.tgz")], capture_output=True, timeout=600)
    finally:
        # terminate on every path, including exceptions and KeyboardInterrupt
        terminate("output fetched" if fetched else "no/partial output — stopping the burn anyway")
        u, b, r_ = rp.spend()
        print(f"final: spend ${u:.2f} | balance ${b:.2f} | burn ${r_}/hr", flush=True)


if __name__ == "__main__":
    main()
