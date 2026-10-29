"""Read exact storage capacity without importing any training dependencies."""
import json
import os
import time
from disk_supervisor import snapshot
result = snapshot(['/', '/tmp', '/home/jovyan/shares/SR006.nfs3'], pending_bytes=3_000_000_000, write_path='/tmp')
result['timestamp_unix'] = time.time()
result['bootstrap_logs_dir'] = '/home/jovyan/shares/SR006.nfs3/mlsub-logs'
result['nfs2_used'] = False
print('STORAGE_PREFLIGHT ' + json.dumps(result), flush=True)
raise SystemExit(0 if result['status'] == 'ok' else 74)
