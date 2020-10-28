"Booster":"gbtree", "booster"
"objective"
"eval_metric"

* "eta": 0.1 : 0.->1.
*"max_depth":4 1->max
*"subsample": 0.9 0.->1.
*"min_child_weight": 2 , 0->max
"seed": 42 
"seed_per_iteration": true
"tree_method": "hist"

"gamma": 0. 0.->max

"nthread": 8
"silent": 1

"base_score": 0.5
"max_delta_step": 0. 0.->max
"alpha": 0. 0.->max
"lambda": 1 0->max
"sketch_eps": 0.03, 0.->1.
"max_leaves": 0  -1 ?
"refresh_leaf": 1 {0,1} ?
"scale_pos_weight": 1. 0.->max
"colsample_bytree": 1 0.->1
"colsample_bynode":1. 0.->1.
"colsample_bylevel": 1. 0->1.
"num_parallel_tree": 1, 1-1 ?cIntVariant(1), 1, gUnknownIndex, nullptr, "num_parallel_tree");
