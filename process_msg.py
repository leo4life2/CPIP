# import numpy as np
# import math
# import os
# import cv2
# import msgpack
# from scipy.spatial.transform import Rotation as R
# import rdp

# def get_orientation(mp1, vp1):
#     def nn_line(xx, yy):
#         points = np.vstack((xx, yy)).T
#         tolerance = 1
#         min_angle = np.pi * 0.22
#         simplified = np.array(rdp.rdp(points.tolist(), tolerance))
#         sx, sy = simplified.T
#         directions = np.diff(simplified, axis=0)
#         theta = angle(directions)
#         idx = np.where(theta > min_angle)[0] + 1
#         org_idx = []
#         for i in range(idx.size):
#             mindist = np.inf
#             minidx = 0
#             for j in range(xx.size):
#                 d = math.dist([sx[idx[i]], sy[idx[i]]], [xx[j], yy[j]])
#                 if d < mindist:
#                     mindist = d
#                     minidx = j
#             org_idx.append(minidx)
#         return xx, yy, theta, org_idx

#     def angle(dir):
#         dir2 = dir[1:]
#         dir1 = dir[:-1]
#         return np.arccos((dir1 * dir2).sum(axis=1) / (np.sqrt((dir1**2).sum(axis=1) * (dir2**2).sum(axis=1))))

#     def key_img(msg_path, video_path):
#         with open(msg_path, "rb") as f:
#             u = msgpack.Unpacker(f)
#             msg = u.unpack()

#         key_frames = msg["keyframes"]
#         print("Point cloud has {} points.".format(len(key_frames)))
#         key_frame = {int(k): v for k, v in key_frames.items()}

#         video_name = video_path.split("/")[-1][:-4]
#         if not os.path.exists(video_name):
#             os.mkdir(video_name)

#         vidcap = cv2.VideoCapture(video_path)
#         fps = int(vidcap.get(cv2.CAP_PROP_FPS)) + 1
#         count = 0

#         tss = []
#         keyfrm_points = []
#         for key in sorted(key_frame.keys()):
#             point = key_frame[key]
#             trans_cw = np.matrix(point["trans_cw"]).T
#             rot_cw = R.from_quat(point["rot_cw"]).as_matrix()

#             rot_wc = rot_cw.T
#             trans_wc = -rot_wc * trans_cw
#             keyfrm_points.append((trans_wc[0, 0], trans_wc[1, 0], trans_wc[2, 0]))
#             vidcap.set(cv2.CAP_PROP_POS_FRAMES, fps * float(point["ts"]))
#             tss.append(point["ts"])

#             success, image = vidcap.read()
#             if not success:
#                 print("capture failed")
#             else:
#                 cv2.imwrite(os.path.join(video_name, str(count) + ".jpg"), image)
#             count += 1
#         keyfrm_points = np.array(keyfrm_points)
#         keyfrm_points = np.delete(keyfrm_points, 1, 1)
#         return keyfrm_points, tss

#     kp1, ts1 = key_img(mp1, vp1)
#     x1, y1, theta1, id1 = nn_line(kp1[:, 0], kp1[:, 1])
#     if len(id1) == 0:
#         id1.append(len(kp1) - 3)

#     return x1, y1, theta1, id1
