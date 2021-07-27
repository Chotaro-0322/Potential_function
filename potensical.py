# インクルードするものはこんな感じ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.rcsetup as rcsetup
import matplotlib
import open3d as o3d
from tqdm import tqdm

from mayavi import mlab

# print("matplotlibrc is ", matplotlib.matplotlib_fname())
# print("matplotlib backend is ", rcsetup.all_backends)


class Potential_function():
  def __init__(self):
    # pcdの読み込み
    pcd = o3d.io.read_point_cloud("./shintoshin_210609_2.pcd")
    print("pcd data is ", pcd)
    pcd_down_samp = pcd.voxel_down_sample(voxel_size=0.1)
    print("pcd_down_samp ", pcd_down_samp)

    # DBSCANクラスタリング
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
      labels = np.array(pcd_down_samp.cluster_dbscan(eps=0.4, min_points=25, print_progress=True))
    print("labels num is ", labels.max() + 1)
    print("labels is ", labels.shape)
    print("labels is ", labels)
    print("labels type is ", type(labels))
    print("big cluster is ", np.where(labels > 0)[0])
    pcd_np = np.asarray(pcd_down_samp.points)
    pcd_np = pcd_np[np.where(labels > 0), :].squeeze()

    print("pcd_np is ", pcd_np.shape)

    # Test
    # pcd_np = pcd_np[pcd_np[:, 2] < 100]
    # pcd_np = pcd_np[pcd_np[:, 2] > -200]
    
    # x軸方向にマップが傾いていたので, アフィン変換により修正
    pcd_np = self.affine(pcd_np, 2.0)

    #　障害物と判断する高さの範囲
    self.low_z = 2.0
    self.upper_z = 2.8

    self.pcd_copy = np.round(pcd_np.copy(), decimals=1)
    pcd_np = pcd_np[pcd_np[:, 2] > self.low_z] # z > 1mの場所のみを取り出す
    pcd_np = pcd_np[pcd_np[:, 2] < self.upper_z] # z < 3mの場所のみを取り出す
    print("pcd_np is ", pcd_np)
    print("pcd_np is ", pcd_np.shape)
    pcd_np = np.round(pcd_np, decimals=1)
    print("pcd_np round is ", pcd_np)

    self.x_range = [self.pcd_copy[:, 0].min(), self.pcd_copy[:, 0].max()]
    self.y_range = [self.pcd_copy[:, 1].min(), self.pcd_copy[:, 1].max()]
    self.z_range = [self.pcd_copy[:, 2].min(), self.pcd_copy[:, 2].max()]
    self.z_limit_range = [pcd_np[:, 2].min(), pcd_np[:, 2].max()]
    print("range_x is ", self.x_range)
    print("range_y is ", self.y_range)
    print("range_z is ", self.z_range)
    # test 大きさを10分の1へ
    self.x_range[0] = np.round(self.x_range[0] / 5, decimals=1)
    self.x_range[1] = np.round(self.x_range[1] / 5, decimals=1)
    self.y_range[0] = np.round(self.y_range[0] / 5, decimals=1)
    self.y_range[1] = np.round(self.y_range[1] / 5, decimals=1)

    # スタートとゴール
    self.start_x, self.start_y   =  0, 0
    self.goal_x , self.goal_y    = self.x_range[1], self.y_range[1]
    # 障害物の座標(何個でも可)
    self.obst = pcd_np[:, :2]
    print("obst is ", self.obst)
    print("obst is ", self.obst.shape)
    # obst = []
    # for i in range(40):
    #   obst.append([30, i])
    obst_x = []
    obst_y = []
    for i in range(len(self.obst)):
      obst_x.append(self.obst[i][0])
      obst_y.append(self.obst[i][1])

    #微分と進むスピード
    delt  = 0.1
    speed = 1
    #障害物とゴールの重みづけ
    self.weight_obst, self.weight_goal = 10, 0
    #ポテンシャルの最大値、最小値
    self.potential_max, self.potential_min = 100, -100
    #障害物と認識する距離
    self.minimum_dist = 3
    #取得間隔(m)
    self.interval = 0.1


  def affine(self, pcd_points, rotate):
    pcd_x = pcd_points[:, 0]
    pcd_y = pcd_points[:, 1] * np.cos(np.deg2rad(rotate)) - pcd_points[:, 2] * np.sin(np.deg2rad(rotate))
    pcd_z = pcd_points[:, 1] * np.sin(np.deg2rad(rotate)) + pcd_points[:, 2] * np.cos(np.deg2rad(rotate))
    print("pcd_x", pcd_x)
    print("pcd_y", pcd_y)
    print("pcd_z", pcd_z)
    affined_pcd = np.stack([pcd_x, pcd_y, pcd_z], 1)
    print("affined pcd is ", affined_pcd)

    return affined_pcd

  def obst_cal(self, x,y,obst):
    tmp_x, tmp_y, tmp = 0, 0, 0
    dist = []
    obst_in_x_y = []
    
    # 障害物と認識する距離以内に計算するオブジェクトを絞り込む
    in_range_x = np.where((obst[:, 0] > x - self.minimum_dist) & (obst[:, 0] < x + self.minimum_dist))[0]
    in_range_y = np.where((obst[:, 1] > y - self.minimum_dist) & (obst[:, 1] < y + self.minimum_dist))[0]
    dobuli_x_y = np.intersect1d(in_range_x, in_range_y)
    obst = obst[dobuli_x_y]

    # print("x is ", x, "y is ", y, "obst is ", obst[0])
    obst_distance = np.sqrt(np.power((obst[:, 0] - x), 2) + np.power((obst[:, 1] - y), 2))
    # print("obst_distance is ", obst_distance[0])
    num_near_obj = np.where(obst_distance < self.minimum_dist)
    # print("num_near_obj is ", num_near_obj)
    near_obj = obst[num_near_obj]
    print("print obst_distance ", obst_distance)
    # print("near_obj.all is ", near_obj)
    # print("near_obj.all is ", near_obj.size == 0)
    # print("near_obj.all is ", near_obj)
    if not near_obj.size == 0:
      # 近くのオブジェクトの中でも最も近いオブジェクトを取り出す.
      # print("num_near_obj is ", obst_distance[num_near_obj])
      min_pos = np.where(obst_distance[num_near_obj].min())
      # print("min_pos is ", min_pos)
      # print("near_obj is ", np.round(near_obj[min_pos].squeeze(), decimals=1))
      return np.round(near_obj[min_pos].squeeze(), decimals=1)
    else:
      return near_obj
  
  def search_range_obj(self, x, y, obst):
    # 障害物と認識する距離以内に計算するオブジェクトを絞り込む
    in_range = np.where((obst[:, 0] > x - self.minimum_dist) & (obst[:, 0] < x + self.minimum_dist)
                        & (obst[:, 1] > y - self.minimum_dist) & (obst[:, 1] < y + self.minimum_dist))[0]
    obst = obst[in_range]
    return obst


  # ポテンシャル関数の計算
  def cal_pot(self, x, y, obst_target):
    tmp_pot = 0
    # print("obst_target is ", obst_target)
    # print("obst_is", obst_target == [])

    # 障害物がないとき(Noneがはいっている)
    if obst_target.size == 0:
      # print("in the if")
      obst_pot = 0
    # 障害物の座標のpotentialはmax
    elif obst_target.size == 1:
      if (obst_target[0] == x) & (obst_target[1] == y):
        obst_pot = self.potential_max
      else:
        # print("obst_is", obst_target)
        obst_pot =  1 / np.sqrt(np.power((x - obst_target[0]), 2) + np.power((y - obst_target[1]), 2))
        obst_pot += obst_pot * self.weight_obst
    else:
      # print("obst_target is ", np.sqrt(np.power((obst_target[:, 0] - x), 2) + np.power((obst_target[:, 1] - y), 2)))
      obst_pot =  1 / np.sqrt(np.power((obst_target[:, 0] - x), 2) + np.power((obst_target[:, 1] - y), 2))
      obst_pot = np.sqrt(np.sum(np.power(obst_pot, 2)))
      obst_pot = obst_pot * self.weight_obst
    
    tmp_pot += obst_pot

    # ゴールの座標はpotentialはmin
    if self.goal_x == x and self.goal_y == y:
      goal_pot = self.potential_min
    else:
      goal_pot = -1 / np.sqrt(pow((x - self.goal_x),  2) + pow((y - self.goal_y),  2))

    pot_all = tmp_pot + self.weight_goal * goal_pot

    return pot_all

  def cal_potential_field(self):
    pot = []
    print("x_range[0] *10 is ", self.x_range[0]*10)
    print("x_range[1]+1 * 10is ", (self.x_range[1])*10 + 1)
    print("y_range[0] *10 is ", self.y_range[0]*10)
    print("y_range[1]+1 * 10is ", (self.y_range[1])*10 + 1)
    # 予め, 座標ごとの物体を検出するために使用するの範囲を計算しておく(計算の高速化のため).
    x_plot, y_plot = np.meshgrid(np.arange(Potential.x_range[0], Potential.x_range[1] + 0.1, 0.1),np.arange(Potential.y_range[0], Potential.y_range[1] + 0.1, 0.1))
    x_prepare_min_range = x_plot - self.minimum_dist
    x_prepare_max_range = x_plot + self.minimum_dist
    y_prepare_min_range = y_plot - self.minimum_dist
    y_prepare_max_range = y_plot + self.minimum_dist
    prepare_range = np.stack([x_prepare_min_range, x_prepare_max_range, y_prepare_min_range, y_prepare_max_range])
    print("prepare_range", prepare_range.shape)

    for y_for_pot in tqdm(range(int(self.y_range[0] * 10), int(self.y_range[1]* 10)+1)):
      y_for_pot /= 10
      # print("y_for_pot is ", y_for_pot)
      tmp_pot = []
      for x_for_pot in range(int(self.x_range[0] * 10), int(self.x_range[1]* 10)+1):
        x_for_pot /= 10

        # 近くの一番近い障害物の座標を特定
        # obst_x_y = self.obst_cal(x_for_pot, y_for_pot, self.obst)
        # 近くの障害物すべての座標を特定
        obst_x_y = self.search_range_obj(x_for_pot, y_for_pot, self.obst)

        potential = self.cal_pot(x_for_pot, y_for_pot, obst_x_y)
        #max,minの範囲内にする
        if potential > self.potential_max:
          potential = self.potential_max
        elif potential < self.potential_min:
          potential = self.potential_min

        tmp_pot.append(potential)
        # print("x is ", x_for_pot, "tmp_pot len is ", len(tmp_pot))
      pot.append(tmp_pot)

    pot = np.array(pot)

    # potを正規化
    pot_mean = pot.mean()
    pot_std = np.std(pot)
    pot = (pot - pot_mean)/pot_std
    print("pot is ", pot.shape)
    return pot


  #ルートをdfに代入
  # def cal_route(self, x, y, df):
  #   count = 0
  #   while True:
  #     count += 1
  #     #対象となる障害物の座標を代入
  #     obst_x_y = obst_cal(x,y)
  #     obst_target_x = obst_x_y[0]
  #     obst_target_y = obst_x_y[1]

  #   #ポテンシャル場を偏微分して，xとy合成
  #     vx = -(cal_pot(x + delt, y, obst_target_x, obst_target_y) - cal_pot(x, y, obst_target_x, obst_target_y)) / delt
  #     vy = -(cal_pot(x, y+delt, obst_target_x, obst_target_y) - cal_pot(x, y, obst_target_x, obst_target_y)) / delt

  #     v = math.sqrt(vx * vx + vy * vy)

  #     # 正規化
  #     vx /= v / speed
  #     vy /= v / speed

  #     # 進める
  #     x += vx
  #     y += vy

  #     # Series型でdfに追加
  #     tmp = pd.Series([x, y, vx, vy, obst_target_x, obst_target_y], index = df.columns)
  #     #print("tmp is ", tmp)
  #     df = df.append(tmp, ignore_index = True) 

  #     # ゴールに近づいた場合，10,000回ループした場合，終了
  #     if goal_x - x < 0.1 and goal_y - y < 0.1:
  #       break
  #     if count > 1000:
  #       break
  #     return df

  #ルートグラフ化
  # def plot_route(self, df):
  #     plt.scatter(df['x'],df['y'])
  #     #スタート、ゴール、障害物をプロット
  #     plt.plot(start_x  , start_y  , marker = 's', color = 'b', markersize = 15)
  #     plt.plot(goal_x   , goal_y   , marker = 's', color = 'b', markersize = 15)
  #     for i in range(len(obst)):
  #         plt.plot(obst_x[i], obst_y[i], marker = 's', color = 'r', markersize = 10)

  #     #print("plot df is ", df)

  #     plt.xlim([x_min, x_max])
  #     plt.ylim([y_min, y_max])
  #     plt.show()

  #ポテンシャル場グラフ化
  def plot3d(self, U,xm,ym, pcd_cord):
      # print("U is ", U.shape)
      # print("xm is ", xm.shape)
      # print("ym is ", ym.shape)
      # # グラフ表示の設定
      # #plt.figure()
      # fig = plt.figure(figsize=(6,6))
      # ax = fig.add_subplot(111, projection="3d")
      # ax.set_box_aspect((1, 1, 0.1))
      # ax.view_init(elev=45, azim=45)
      # ax.tick_params(labelsize=7)    # 軸のフォントサイズ
      # ax.set_xlabel("x", fontsize=10)
      # ax.set_ylabel("y", fontsize=10)
      # ax.set_zlabel("U", fontsize=10)
      # ax.set_xlim(-200, 200)
      # ax.set_ylim(-300, 200)
      # ax.set_zlim(-10, 390)
      # print("xm\n", xm, "ym\n", ym)
      print("pcd_cord is\n", pcd_cord[:, 0].shape)
      print("pcd_cord is\n", pcd_cord[:, 1].shape)
      print("pcd_cord is\n", pcd_cord[:, 2].shape)
      print("pcd_cord xm is\n", xm.shape)
      print("pcd_cord ymis\n", ym.shape)
      print("pcd_cord U is\n", U.shape)
      # ax.pbaspect = [1, 1, 0.25]
      # surf = ax.plot_surface(xm, ym, U, rstride=1, cstride=1, cmap=cm.coolwarm)
      # scat = ax.scatter(pcd_cord[:, 0], pcd_cord[:, 1], pcd_cord[:, 2], s = 0.001, alpha=0.2)
      # plt.show()

      #mayavi
      pcd_onzero = pcd_cord[(pcd_cord[:, 2] >= self.low_z) & (pcd_cord[:, 2] < self.upper_z)]
      pcd_underzero = pcd_cord[pcd_cord[:, 2] < self.low_z]
      pcd_onlimit = pcd_cord[pcd_cord[:, 2] >= self.upper_z]
      mlab.points3d(pcd_onzero[:, 0], pcd_onzero[:, 1], pcd_onzero[:, 2], scale_factor=0.1, color=(0, 1, 0))
      mlab.points3d(pcd_underzero[:, 0], pcd_underzero[:, 1], pcd_underzero[:, 2], scale_factor=0.1, color=(1, 1, 1))
      mlab.points3d(pcd_onlimit[:, 0], pcd_onlimit[:, 1], pcd_onlimit[:, 2], scale_factor=0.1, color=(1, 1, 1))
      U -= 1
      surf = mlab.mesh(xm, ym, U, colormap="cool")
      lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
      lut[:, -1] = np.linspace(50, 100, 256)
      surf.module_manager.scalar_lut_manager.lut.table = lut

      mlab.show()

Potential = Potential_function()
pot = Potential.cal_potential_field()
print("Potential.np.arange x_range is ", np.arange(Potential.x_range[0], Potential.x_range[1] + 0.1, 0.1))
print("Potential.np.arange y_range is ", np.arange(Potential.y_range[0], Potential.y_range[1] + 0.1, 0.1))
x_plot, y_plot = np.meshgrid(np.arange(Potential.x_range[0], Potential.x_range[1] + 0.1, 0.1),np.arange(Potential.y_range[0], Potential.y_range[1] + 0.1, 0.1))
Potential.plot3d(pot, x_plot, y_plot, Potential.pcd_copy)

# df = pd.DataFrame(columns=['x','y','vx','vy','obst_x','obst_y'])
# df = cal_route(start_x, start_y, df)
# plot_route(df)