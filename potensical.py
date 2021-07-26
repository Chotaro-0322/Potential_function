# インクルードするものはこんな感じ
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import open3d as o3d
from tqdm import tqdm


class Potential_function():
  def __init__(self):
    # pcdの読み込み
    pcd = o3d.io.read_point_cloud("./shintoshin_210609_2.pcd")
    print("pcd data is ", pcd)
    pcd_np = np.asarray(pcd.points)
    self.pcd_copy = pcd_np.copy()
    pcd_np = pcd_np[pcd_np[:, 2] > 1] # z > 1mの場所のみを取り出す
    pcd_np = pcd_np[pcd_np[:, 2] < 3] # z < 3mの場所のみを取り出す
    print("pcd_np is ", pcd_np)
    print("pcd_np is ", pcd_np.shape)

    self.x_range = [pcd_np[:, 0].min().astype(np.int32), pcd_np[:, 0].max().astype(np.int32)]
    self.y_range = [pcd_np[:, 1].min().astype(np.int32), pcd_np[:, 1].max().astype(np.int32)]
    self.z_range = [self.pcd_copy[:, 2].min().astype(np.int32), self.pcd_copy[:, 2].max().astype(np.int32)]
    self.z_limit_range = [pcd_np[:, 2].min().astype(np.int32), pcd_np[:, 2].max().astype(np.int32)]
    print("range_max is ", self.x_range)
    print("range_min is ", self.y_range)
    # test 大きさを10分の1へ
    # self.x_range[0] = int(self.x_range[0] / 10)
    # self.x_range[1] = int(self.x_range[1] / 10)
    # self.y_range[0] = int(self.y_range[0] / 10)
    # self.y_range[1] = int(self.y_range[1] / 10)

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
    self.weight_obst, self.weight_goal = 0.1, 10
    #ポテンシャルの最大値、最小値
    self.potential_max, self.potential_min = 1, -1
    #障害物と認識する距離
    self.minimum_dist = 10
    #取得間隔(m)
    self.interval = 0.1

  def obst_cal(self, x,y,obst):
    tmp_x, tmp_y, tmp = 0, 0, 0
    dist = []
    obst_in_x_y = []
    
    # print("x is ", x, "y is ", y, "obst is ", obst[0])
    obst_distance = np.sqrt(np.power((obst[:, 0] - x), 2) + np.power((obst[:, 1] - y), 2))
    # print("obst_distance is ", obst_distance[0])
    num_near_obj = np.where(obst_distance < self.minimum_dist)
    # print("num_near_obj is ", num_near_obj)
    near_obj = obst[num_near_obj]
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
    elif (obst_target[0] == x) & (obst_target[1] == y):
      obst_pot = self.potential_max
    else:
      # print("obst_is", obst_target)
      obst_pot =  1 / math.sqrt(pow((x - obst_target[0]), 2) + pow((y - obst_target[1]), 2))
      obst_pot += obst_pot * self.weight_obst

    tmp_pot += obst_pot

    # ゴールの座標はpotentialはmin
    if self.goal_x == x and self.goal_y == y:
      goal_pot = self.potential_min
    else:
      goal_pot = -1 / math.sqrt(pow((x - self.goal_x),  2) + pow((y - self.goal_y),  2))

    pot_all    = tmp_pot + self.weight_goal * goal_pot

    return pot_all

  def cal_potential_field(self):
    pot = []
    print("y_range is ", self.y_range)
    for y_for_pot in tqdm(range(self.y_range[0], self.y_range[1] + 1)):
      tmp_pot = []
      for x_for_pot in range(self.x_range[0], self.x_range[1] + 1):
        potential = 0

        # 対象となる障害物の座標を代入
        obst_x_y = self.obst_cal(x_for_pot, y_for_pot, self.obst)

        potential += self.cal_pot(x_for_pot, y_for_pot, obst_x_y)
        #max,minの範囲内にする
        if potential > self.potential_max:
          potential = self.potential_max
        elif potential < self.potential_min:
          potential = self.potential_min

        tmp_pot.append(potential)
      pot.append(tmp_pot)

    pot = np.array(pot)
    print("pot is ", pot.shape)
    return pot


  #ルートをdfに代入
  def cal_route(self, x, y, df):
    count = 0
    while True:
      count += 1
      #対象となる障害物の座標を代入
      obst_x_y = obst_cal(x,y)
      obst_target_x = obst_x_y[0]
      obst_target_y = obst_x_y[1]

    #ポテンシャル場を偏微分して，xとy合成
      vx = -(cal_pot(x + delt, y, obst_target_x, obst_target_y) - cal_pot(x, y, obst_target_x, obst_target_y)) / delt
      vy = -(cal_pot(x, y+delt, obst_target_x, obst_target_y) - cal_pot(x, y, obst_target_x, obst_target_y)) / delt

      v = math.sqrt(vx * vx + vy * vy)

      # 正規化
      vx /= v / speed
      vy /= v / speed

      # 進める
      x += vx
      y += vy

      # Series型でdfに追加
      tmp = pd.Series([x, y, vx, vy, obst_target_x, obst_target_y], index = df.columns)
      #print("tmp is ", tmp)
      df = df.append(tmp, ignore_index = True) 

      # ゴールに近づいた場合，10,000回ループした場合，終了
      if goal_x - x < 0.1 and goal_y - y < 0.1:
        break
      if count > 1000:
        break
      return df

  #ルートグラフ化
  def plot_route(self, df):
      plt.scatter(df['x'],df['y'])
      #スタート、ゴール、障害物をプロット
      plt.plot(start_x  , start_y  , marker = 's', color = 'b', markersize = 15)
      plt.plot(goal_x   , goal_y   , marker = 's', color = 'b', markersize = 15)
      for i in range(len(obst)):
          plt.plot(obst_x[i], obst_y[i], marker = 's', color = 'r', markersize = 10)

      #print("plot df is ", df)

      plt.xlim([x_min, x_max])
      plt.ylim([y_min, y_max])
      plt.show()

  #ポテンシャル場グラフ化
  def plot3d(self, U,xm,ym, pcd_cord):
      print("U is ", U.shape)
      print("xm is ", xm.shape)
      print("ym is ", ym.shape)
      # グラフ表示の設定
      plt.figure()
      fig = plt.figure(facecolor="w", figsize=plt.figaspect(0.25))
      ax = fig.add_subplot(111, projection="3d")
      ax.tick_params(labelsize=7)    # 軸のフォントサイズ
      ax.set_xlabel("x", fontsize=10)
      ax.set_ylabel("y", fontsize=10)
      ax.set_zlabel("U", fontsize=10)
      print("xm\n", xm, "ym\n", ym)
      print("pcd_cord is\n", pcd_cord[:, 0].shape)
      print("pcd_cord is\n", pcd_cord[:, 1].shape)
      print("pcd_cord is\n", pcd_cord[:, 2].shape)
      ax.pbaspect = [self.x_range[1]-self.x_range[0], self.y_range[1]-self.y_range[0], self.z_range[1]-self.z_range[0]]
      surf = ax.plot_surface(xm, ym, U, rstride=1, cstride=1, cmap=cm.coolwarm)
      scat = ax.scatter(pcd_cord[:, 0], pcd_cord[:, 1], pcd_cord[:, 2], s = 0.5)
      plt.show()

Potential = Potential_function()
pot = Potential.cal_potential_field()
x_plot, y_plot = np.meshgrid(np.arange(Potential.x_range[0], Potential.x_range[1] + 1),np.arange(Potential.y_range[0], Potential.y_range[1] +1))
Potential.plot3d(pot, x_plot, y_plot, Potential.pcd_copy)

# df = pd.DataFrame(columns=['x','y','vx','vy','obst_x','obst_y'])
# df = cal_route(start_x, start_y, df)
# plot_route(df)