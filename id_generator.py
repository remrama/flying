import calendar
import random
import uuid


rd = random.Random()
rd.seed(32)

months = [x.lower() for x in calendar.month_abbr[1:]]

export_path = "./ids.csv"
n = 6876

ids = []

while len(ids) < n:
    h = uuid.UUID(int=rd.getrandbits(128)).hex[:4]
    if (
        h.isidentifier()
        and any(x.isnumeric() for x in h)
        and h not in ids
        and not any(x in h for x in months)
        ):
        ids.append(h)

ids = [f"fly-{x}" for x in ids]
with open(export_path, "w", encoding="utf-8") as f:
    f.writelines("\n".join(ids))