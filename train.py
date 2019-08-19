import yaml
from pprint import pformat
from training import run_training


CONFIG = yaml.load(open('CONFIG.yaml'), Loader=yaml.FullLoader)
RESULTS = run_training(CONFIG)
print(pformat(RESULTS))

