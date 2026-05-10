import sys
import json

sys.path.append('/app')
from tools.surgeon_tools import AudiophileUpsamplerTool

try:
    tool = AudiophileUpsamplerTool()
    print('Running tool...')
    result = tool._run('/data/input/C.wav', '/data/intermediate/C.wav_96k.wav')
    print('RESULT:')
    print(result)
except Exception as e:
    print('FATAL ERROR:', str(e))
