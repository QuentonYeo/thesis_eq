import seisbench
from seisbench import util

import seisbench.models as sbm
print("Available PhaseNet weights:", sbm.PhaseNet.list_pretrained(remote=True))  # may still warn
m = sbm.PhaseNet.from_pretrained("stead", update=True)

# print(seisbench._version_)
# seisbench.remote_root = "https://hifis-storage.desy.de:2880/Helmholtz/HelmholtzAI/SeisBench/"
# models_url = seisbench.remote_root + "models/v3/"
# data_url   = seisbench.remote_root + "datasets/"

# print("Models dir:", util.ls_webdav(models_url, precheck_timeout=0))  # disables the HEAD precheck
# print("Datasets dir:", util.ls_webdav(data_url, precheck_timeout=0))
