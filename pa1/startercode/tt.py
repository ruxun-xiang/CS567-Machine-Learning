import numpy as np
features = [[3, 4], [1, -1], [0, 0]]
normalized_features = []

np.set_printoptions(precision=6)
for i in range(len(features)):
    feature = np.array(features[i])
    norm = np.linalg.norm(feature)
    if norm != 0:
        normalized_feature = feature / norm
        normalized_feature = np.around(normalized_feature.tolist(), 6)
        normalized_features.append(normalized_feature.tolist())
    else:
        normalized_features.append(0)
print(normalized_features)

# np_features = np.array(features)
# col_max = np.max(np_features, axis=0)
# col_min = np.min(np_features, axis=0)
#
# for feature in features:
#     # print(feature)
#     for i in range(len(feature)):
#         # print(feature[i])
#         if col_min[i] != col_max[i]:
#             feature[i] = (feature[i] - col_min[i]) / (col_max[i] - col_min[i])
#             feature[i] = np.around(feature[i], 5)
#         else:
#             feature[i] = 0
# print(features)


