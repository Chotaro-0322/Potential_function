import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.rcsetup as rcsetup
import matplotlib
import open3d as o3d
from tqdm import tqdm
import csv
import json
import yaml

from mayavi import mlab

class Potential_function():
  def __init__(self, cfg):
    print("cfg is ", cfg)
    
    #取得間隔(m)
    self.sampling = cfg["valid_pcd_interval"]
    self.eps = cfg["obscan_cls"]["eps"]
    self.min_points = cfg["obscan_cls"]["min_points"]
    self.num_interval_grid = int(1 / self.sampling)
    self.decimals = len(str(int(self.num_interval_grid))) - 1

    self.low_z = cfg["low_z"]
    self.upper_z = cfg["upper_z"]
    self.affine_x_rot = cfg["affine_x_rot"]

    self.waypoints = self.read_csv_waypoint(cfg["csv_waypoint"])

    #  pcdの読み込み
    self.pcd_np, self.pcd_copy = self.read_pcd_process(cfg["pcd_data"])

    self.x_range = [self.pcd_copy[:, 0].min(), self.pcd_copy[:, 0].max()]
    self.y_range = [self.pcd_copy[:, 1].min(), self.pcd_copy[:, 1].max()]
    self.z_range = [self.pcd_copy[:, 2].min(), self.pcd_copy[:, 2].max()]
    self.z_limit_range = [self.pcd_np[:, 2].min(), self.pcd_np[:, 2].max()]
    # test 大きさを10分の1へ
    # self.x_range[0] = np.round(self.x_range[0] / 5, decimals=self.decimals)
    # self.x_range[1] = np.round(self.x_range[1] / 5, decimals=self.decimals)
    # self.y_range[0] = np.round(self.y_range[0] / 5, decimals=self.decimals)
    # self.y_range[1] = np.round(self.y_range[1] / 5, decimals=self.decimals)
    # self.waypoints = np.array([points for points in self.waypoints if (self.x_range[0] < points[0]) 
    #                                                           & (self.x_range[1] > points[0])
    #                                                           & (self.y_range[0] < points[1])
    #                                                           & (self.y_range[1] > points[1])
    #                                                           ])
    #
    # スタートとゴール
    self.start_x, self.start_y   =  cfg["start"]["x"], cfg["start"]["y"]
    
    self.goal_x , self.goal_y    = cfg["goal"]["x"], cfg["goal"]["y"]
    # 障害物の座標(何個でも可)
    self.obst = self.pcd_np[:, :2]

    #微分と進むスピード
    self.delt  = cfg["delt"]
    self.speed = cfg["speed"]
    #障害物とゴールの重みづけ
    self.weight_obst, self.weight_goal = cfg["weight"]["obst"], cfg["weight"]["goal"]
    #ポテンシャルの最大値、最小値
    self.potential_max, self.potential_min = cfg["potential_range"]["max"], cfg["potential_range"]["min"]
    #障害物と認識する距離
    self.minimum_dist = cfg["minimum_dist"]

    self.moving_avg = cfg["moving_avg"]

  def read_pcd_process(self, pcd_name):
    pcd = o3d.io.read_point_cloud(pcd_name)
    pcd_down_samp = pcd.voxel_down_sample(voxel_size=self.sampling)

    # DBSCANクラスタリング
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd_down_samp.cluster_dbscan(eps=self.eps,
                                                       min_points=self.min_points, 
                                                       print_progress=True))
    pcd_np = np.asarray(pcd_down_samp.points)
    pcd_np = pcd_np[np.where(labels > 0), :].squeeze()

    # x軸方向にマップが傾いていたので, アフィン変換により修正
    pcd_np = self.affine(pcd_np, self.affine_x_rot)

    pcd_copy = np.round(pcd_np.copy(), decimals = self.decimals)
    
    pcd_np = pcd_np[pcd_np[:, 2] > self.low_z] # z > 1mの場所のみを取り出す
    pcd_np = pcd_np[pcd_np[:, 2] < self.upper_z] # z < 3mの場所のみを取り出す
    pcd_np = np.round(pcd_np, decimals=self.decimals)
    return pcd_np, pcd_copy
    
  def read_csv_waypoint(self, csv_name):
      df_waypoints = pd.read_csv(csv_name)
      waypoints = df_waypoints.loc[:, ["x", "y", "z"]].values
      waypoints = self.affine(waypoints, self.affine_x_rot)
      waypoints = np.round(waypoints, decimals=self.decimals)
      return waypoints

  def affine(self, pcd_points, rotate : float):
    pcd_x = pcd_points[:, 0]
    pcd_y = pcd_points[:, 1] * np.cos(np.deg2rad(rotate)) - pcd_points[:, 2] * np.sin(np.deg2rad(rotate))
    pcd_z = pcd_points[:, 1] * np.sin(np.deg2rad(rotate)) + pcd_points[:, 2] * np.cos(np.deg2rad(rotate))
    affined_pcd = np.stack([pcd_x, pcd_y, pcd_z], 1)

    return affined_pcd

  def obst_cal(self, x,y,obst):
    # 障害物と認識する距離以内に計算するオブジェクトを絞り込む
    in_range_x = np.where((obst[:, 0] > x - self.minimum_dist) & (obst[:, 0] < x + self.minimum_dist))[0]
    in_range_y = np.where((obst[:, 1] > y - self.minimum_dist) & (obst[:, 1] < y + self.minimum_dist))[0]
    dobuli_x_y = np.intersect1d(in_range_x, in_range_y)
    obst = obst[dobuli_x_y]

    obst_distance = np.sqrt(np.power((obst[:, 0] - x), 2) + np.power((obst[:, 1] - y), 2))
    num_near_obj = np.where(obst_distance < self.minimum_dist)
    near_obj = obst[num_near_obj]
    if not near_obj.size == 0:
      # 近くのオブジェクトの中でも最も近いオブジェクトを取り出す.
      min_pos = np.where(obst_distance[num_near_obj].min())
      return np.round(near_obj[min_pos].squeeze(), decimals=self.decimals)
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
    # 障害物がないとき(Noneがはいっている)
    if obst_target.size == 0:
      obst_pot = 0
    # 障害物の座標のpotentialはmax
    elif obst_target.size == 1:
      if (obst_target[0] == x) & (obst_target[1] == y):
        obst_pot = self.potential_max
      else:
        obst_pot =  1 / np.sqrt(np.power((x - obst_target[0]), 2) + np.power((y - obst_target[1]), 2))
        obst_pot += obst_pot * self.weight_obst
    else:
      obst_pot =  1 / np.sqrt(np.power((obst_target[:, 0] - x), 2) + np.power((obst_target[:, 1] - y), 2))
      obst_pot = np.sqrt(np.sum(np.power(obst_pot, 2)))
      obst_pot = obst_pot * self.weight_obst

    # ゴールの座標はpotentialはmin
    if self.goal_x == x and self.goal_y == y:
      goal_pot = self.potential_min
    else:
      goal_pot = -1 / np.sqrt(np.power((x - self.goal_x),  2) + np.power((y - self.goal_y),  2))

    pot_all = obst_pot + self.weight_goal * goal_pot

    return pot_all

  def cal_potential_field(self):
    pot = []
    for y_for_pot in tqdm(range(int(self.y_range[0] * self.num_interval_grid), int(self.y_range[1]* self.num_interval_grid)+1)):
      y_for_pot /= self.num_interval_grid
      tmp_pot = []
      for x_for_pot in range(int(self.x_range[0] * self.num_interval_grid), int(self.x_range[1]* self.num_interval_grid)+1):
        x_for_pot /= self.num_interval_grid

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
      pot.append(tmp_pot)

    pot = np.array(pot)

    # potを正規化
    pot_mean = pot.mean()
    pot_std = np.std(pot)
    pot = (pot - pot_mean)/pot_std
    return pot

  def cal_waypoints_weight(self, waypoints, xm, ym, U):
    for i, points in enumerate(waypoints):
      x_pos = np.where(xm[0] == points[0])[0]
      y_pos = np.where(ym.T[0] == points[1])[0]
      if i == 0:
        waypoints_array = U[y_pos, x_pos]
      else:
        waypoints_array = np.append(waypoints_array, U[y_pos, x_pos])
    return waypoints_array

  def waypoint_moving_average(self, waypoint_weight, moving_range):
    pd_weight = pd.Series(waypoint_weight)
    moving_ave = pd_weight.rolling(window=moving_range, min_periods=1, center=True).mean()
    return moving_ave.values
      

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
    """ matplotlib (めちゃくちゃ遅い)
    print("U is ", U.shape)
    print("xm is ", xm.shape)
    print("ym is ", ym.shape)
    # グラフ表示の設定
    #plt.figure()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 0.1))
    ax.view_init(elev=45, azim=45)
    ax.tick_params(labelsize=7)    # 軸のフォントサイズ
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("U", fontsize=10)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-300, 200)
    ax.set_zlim(-10, 390)
    ax.pbaspect = [1, 1, 0.25]
    surf = ax.plot_surface(xm, ym, U, rstride=1, cstride=1, cmap=cm.coolwarm)
    scat = ax.scatter(pcd_cord[:, 0], pcd_cord[:, 1], pcd_cord[:, 2], s = 0.001, alpha=0.2)
    plt.show()
    """
    waypoints_weight = self.cal_waypoints_weight(self.waypoints, xm, ym, U)
    # 移動平均
    waypoints_weight = self.waypoint_moving_average(waypoints_weight, self.moving_avg)
    #mayavi
    pcd_onzero = pcd_cord[(pcd_cord[:, 2] >= self.low_z) & (pcd_cord[:, 2] < self.upper_z)]
    pcd_underzero = pcd_cord[pcd_cord[:, 2] < self.low_z]
    pcd_onlimit = pcd_cord[pcd_cord[:, 2] >= self.upper_z]
    mlab.points3d(pcd_onzero[:, 0], pcd_onzero[:, 1], pcd_onzero[:, 2], scale_factor=0.05, color=(0, 1, 0))
    mlab.points3d(pcd_underzero[:, 0], pcd_underzero[:, 1], pcd_underzero[:, 2], scale_factor=0.1, color=(1, 1, 1))
    mlab.points3d(pcd_onlimit[:, 0], pcd_onlimit[:, 1], pcd_onlimit[:, 2], scale_factor=0.05, color=(1, 1, 1))
    U -= 1
    waypoint_w_min = waypoints_weight.min()
    waypoint_w_max = waypoints_weight.max()
    waypoints_weight = (waypoints_weight - waypoint_w_min) / (waypoint_w_max - waypoint_w_min)
    print("waypoints_weight is ", waypoints_weight)
    # mlab.points3d(self.waypoints[:, 0], self.waypoints[:, 1], waypoints_weight, scale_factor=0.5, colormap="cool")
    [mlab.points3d(waypoint[0], waypoint[1], waypoint_w + 2, scale_factor=0.5, color=(waypoint_w, 0, 1 - waypoint_w)) for waypoint, waypoint_w in zip(self.waypoints, waypoints_weight)]
    surf = mlab.mesh(xm, ym, U, colormap="cool")
    lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:, -1] = np.linspace(50, 100, 256)
    surf.module_manager.scalar_lut_manager.lut.table = lut
    mlab.show()

  def write_json(self, xm, ym, U, json_name):
    pot_dict = {}
    pot_dict["potential_xm"] = xm.tolist()
    pot_dict["potential_ym"] = ym.tolist()
    pot_dict["potential_U"] = U.tolist()
    with open(json_name, mode = "w", encoding="utf-8") as file:
      json.dump(pot_dict, file, ensure_ascii = False, indent=2)
  
  def x_y_decimals(self, xm, ym):
    return np.round(xm, decimals=self.decimals), np.round(ym, decimals=self.decimals)


def main():
  with open("./config.yaml") as file:
    cfg = yaml.safe_load(file)

  Potential = Potential_function(cfg)
  pot = Potential.cal_potential_field()
  x_plot, y_plot = np.meshgrid(np.arange(Potential.x_range[0], Potential.x_range[1] + cfg["valid_pcd_interval"], cfg["valid_pcd_interval"]),
                               np.arange(Potential.y_range[0], Potential.y_range[1] + cfg["valid_pcd_interval"], cfg["valid_pcd_interval"]))
  x_plot, y_plot = Potential.x_y_decimals(x_plot, y_plot)
  Potential.write_json(x_plot, y_plot, pot, cfg["json_data"])
  Potential.plot3d(pot, x_plot, y_plot, Potential.pcd_copy)

  # 経路の最適化
  # df = pd.DataFrame(columns=['x','y','vx','vy','obst_x','obst_y'])
  # df = cal_route(start_x, start_y, df)
  # plot_route(df)

if __name__=="__main__":
    main()