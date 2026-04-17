import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"d:\python_projects\rag\rag-claim-attribution")))

from src.attribution.matrix import SupportMatrix

sm = SupportMatrix()
atomic_facts = ["Who won the Champions League in 2015? Barcelona won."]
evidences = [{"sentence": "Barcelona won the 2015 Champions League."}]

print("Computing...")
matrix = sm.compute(atomic_facts, evidences)
print("Done:", matrix)
