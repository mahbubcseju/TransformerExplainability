# coding=utf-8

import os
import json

from config import Arg
from explain_main import CodebertModel


args = Arg()
model = CodebertModel()
model.load_model(args)


explanations = model.create_explanations(args)


f = open('output.json', 'w')
f.write(json.dumps(explanations))
f.close()
