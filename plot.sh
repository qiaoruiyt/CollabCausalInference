# python main.py --dataset jobs 
# python main.py --dataset tcga 
python main.py --dataset ihdp 

# python main.py --dataset jobs --estimator ipw
# python main.py --dataset jobs --estimator aipw
# python main.py --dataset ihdp --estimator ipw
# python main.py --dataset ihdp --estimator aipw
# python main.py --dataset tcga --estimator ipw
# python main.py --dataset tcga --estimator aipw
# python main.py --dataset jobs --estimator por
# python main.py --dataset ihdp --estimator por
# python main.py --dataset tcga --estimator por
# python main.py --dataset jobs --estimator por --model xgboost
# python main.py --dataset ihdp --estimator por --model xgboost
# python main.py --dataset tcga --estimator por --model xgboost

# python exp_sensitivity.py --dataset ihdp
# python exp_sensitivity.py --dataset jobs
# python exp_sensitivity.py --dataset tcga