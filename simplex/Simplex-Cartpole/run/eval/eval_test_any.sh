#!/usr/bin/env sh
python main_ips.py --mode 'test' --config ./config/simplex/eval/continual_safe_learn4.json \
--weights './models/continual_safe_learn_ep5_best' --id 'test_any'

#python main_ips.py --mode 'test' --config ./config/simplex/eval/continual_unsafe_learn4.json \
#--weights './models/continual_unsafe_learn_ep5_best' --id 'test_any'

