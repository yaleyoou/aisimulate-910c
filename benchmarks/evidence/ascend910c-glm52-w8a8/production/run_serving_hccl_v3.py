"""Run the corrected collector under the existing managed restore protocol."""

import os
import shutil

import run_serving_hccl_nodes as campaign

os.umask(0o022)
campaign.OUTPUT = campaign.ROOT / "hccl-replay-verified-20261022/local"
campaign.OUTPUT.parent.mkdir(exist_ok=True)
shutil.chown(campaign.OUTPUT.parent, user="youxingye", group="youxingye")
campaign.CONTAINER = "aisim-910c-hccl-v3-local-20261022"


if __name__ == "__main__":
    campaign.main()
