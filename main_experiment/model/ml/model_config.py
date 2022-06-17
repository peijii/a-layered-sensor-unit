# =====================================================
# 
#                      SVM
# 
# ================== fusion data ======================
#              best parameter combinations 
#              kernel = 'rbf', C = 9.998, gamma = 0.1
#              acc = 0.8232  f1-score = 0.8205
# =====================================================
#
#
# ======================= semg data ===================
#         best parameter combinations 
#         kernel = 'rbf', C = 6.67, gamma = 0.46415888336127725
#         acc = 0.6201   f1-score = 0.6159
# =====================================================
#
#
# ======================= fmg data ====================
#         best parameter combinations 
#         kernel = 'rbf', C = 9.968, gamma = 0.46415888336127725
#         acc = 0.4772   f1-score = 0.4438
# =====================================================



# =====================================================
# 
#                      RFC
# 
# ================== fusion data ======================
#              best parameter combinations 
#                  n_estimators=172, 
#                  max_depth=22,
#                  min_samples_leaf=1,
#                  min_samples_split=2,
#                  max_features=4,
#                  acc = 0.8012  f1-score = 0.8010
# =====================================================
#
#
# ======================= semg data ===================
#              best parameter combinations 
#                  n_estimators=197,
#                  max_depth=27,
#                  min_samples_leaf=1,
#                  min_samples_split=2,
#                  max_features=3,
#                  acc = 0.5685  f1-score = 0.5623
# =====================================================
#
#
# ======================= fmg data ====================
#         best parameter combinations 
#                  n_estimators=9,
#                  max_depth=27,
#                  min_samples_leaf=1,
#                  min_samples_split=2,
#                  max_features=4,
#                  acc = 0.5212  f1-score = 0.5179
# =====================================================



# =====================================================
# 
#                      XGBOOST
# 
# ================== fusion data ======================
#              best parameter combinations 
#                  n_estimators=182, 
#                  eta=0.25,
#                  max_depth=22,           
#                  acc = 0.7458  f1-score = 0.7419
# =====================================================
#
#
# ======================= semg data ===================
#              best parameter combinations 
#                  n_estimators=197,
#                  eta=0.2667,
#                  max_depth=41,
#                  acc = 0.4884  f1-score = 0.4810
# =====================================================
#
#
# ======================= fmg data ====================
#         best parameter combinations 
#                  n_estimators=9,
#                  max_depth=27,
#                  max_features=6,
#                  acc = 0.4766  f1-score = 0.4668
# =====================================================

# =====================================================
# 
#                      KNN
# 
# ================== fusion data ======================
#              best parameter combinations 
#                  n_neighbors=1, 
#                  p=1,
#                  weights=uniform,           
#                  acc = 0.8044  f1-score = 0.8047
# =====================================================
#
#
# ======================= semg data ===================
#              best parameter combinations 
#                  n_neighbors=4, 
#                  p=2,
#                  weights=distance, 
#                  acc = 0.5567  f1-score = 0.5545
# =====================================================
#
#
# ======================= fmg data ====================
#         best parameter combinations 
#                  n_neighbors=1, 
#                  p=1,
#                  weights=uniform, 
#                  acc = 0.5013  f1-score = 0.5022
# =====================================================