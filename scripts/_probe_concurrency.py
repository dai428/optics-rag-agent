"""并发首屏压力测试：验证 get_vectorstore() 的双重检查锁。

不修之前：3 线程同时进入会抛
  ValueError: Could not connect to tenant default_tenant
修改之后：只构造一次，所有线程拿到同一个对象。
"""

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import knowledge_base as kb  # noqa: E402

N = 6

errors = []


def hit(i):
    try:
        vs = kb.get_vectorstore()
        return id(vs), vs._collection.count()
    except Exception:  # noqa: BLE001
        errors.append(f"thread-{i}:\n{traceback.format_exc()}")
        return None


with ThreadPoolExecutor(max_workers=N) as pool:
    results = list(pool.map(hit, range(N)))

ok = [r for r in results if r]
ids = {r[0] for r in ok}
counts = {r[1] for r in ok}

print(f"线程数        : {N}")
print(f"成功线程      : {len(ok)}/{N}")
print(f"不同实例数    : {len(ids)}  (应为 1)")
print(f"命中子块数    : {counts}")
if errors:
    print(f"报错线程      : {len(errors)}")
    print(errors[0][:1200])
    sys.exit(1)
if len(ids) != 1 or len(ok) != N:
    print("FAIL: 单例或并发语义不成立")
    sys.exit(1)
print("PASS: 并发下仅构造一次向量库")
