import sys
import json
import traceback

sys.path.append('/app')
from tools.surgeon_tools import SOTAMasteringChainTool

try:
    tool = SOTAMasteringChainTool()
    print('Testing SOTA Mastering Chain...')
    res = tool._run('/data/input/C.wav', '/data/output/C_restored.wav')
    print('RESULT:')
    print(res)
except Exception as e:
    print('FATAL EXCEPTION:')
    traceback.print_exc()
