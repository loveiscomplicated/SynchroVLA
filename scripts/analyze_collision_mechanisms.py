"""CLI for guard and near-target contact mechanism audits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_gnn_recurrent.training.collision_mechanism_audit import main


if __name__ == "__main__":
    main()
