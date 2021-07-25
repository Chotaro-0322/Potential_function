# インクルードするものはこんな感じ
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# スタートとゴール
start_x, start_y   =  0, 0
goal_x , goal_y    = 50,50
# 障害物の座標(何個でも可)
obst = [[15,20],[20,20]]
# obst = []
# for i in range(40):
#   obst.append([30, i])
obst_x = []
obst_y = []
for i in range(len(obst)):
  obst_x.append(obst[i][0])
  obst_y.append(obst[i][1])

#微分と進むスピード
delt  = 0.1
speed = 1
#障害物とゴールの重みづけ
weight_obst, weight_goal = 0.1, 10
#それぞれの軸の範囲
x_min, y_min = 0, 0
x_max, y_max = 50, 50
#ポテンシャルの最大値、最小値
potential_max, potential_min = 1, -1
#障害物と認識する距離
minimum_dist = 10

def obst_cal(x,y):
  tmp_x, tmp_y, tmp = 0, 0, 0
  dist = []
  obst_in_x_y = []

  # 障害物の数だけ現在地との距離を計算して最も近いものを返す
  # 閾値以上だったらNone
  for i in range(len(obst)):
    dist.append(math.sqrt((x - obst_x[i]) ** 2 + (y - obst_y[i]) ** 2))
  for i in range(len(obst)): 
    if dist[i] == min(dist):
      tmp_x, tmp_y = obst_x[i],obst_y[i]
      tmp = i
      if dist[i] >= minimum_dist:
        tmp_x, tmp_y = None,None

  obst_in_x_y.append(tmp_x)
  obst_in_x_y.append(tmp_y)
  return obst_in_x_y

  # ポテンシャル関数の計算
def cal_pot(x, y, obst_target_x, obst_target_y):
  tmp_pot = 0

  # 障害物がないとき(Noneがはいっている)
  if obst_target_x == None or obst_target_y == None:
    obst_pot = 0

  # 障害物の座標のpotentialはmax
  elif obst_target_x == x and obst_target_y == y:
    obst_pot = potential_max
  else:
    obst_pot =  1 / math.sqrt(pow((x - obst_target_x), 2) + pow((y - obst_target_y), 2))
    obst_pot += obst_pot * weight_obst

  tmp_pot += obst_pot

  # ゴールの座標はpotentialはmin
  if goal_x == x and goal_y == y:
    goal_pot = potential_min
  else:
    goal_pot = -1 / math.sqrt(pow((x - goal_x),  2) + pow((y - goal_y),  2))

  pot_all    = tmp_pot + weight_goal * goal_pot

  return pot_all

#ルートをdfに代入
def cal_route(x, y, df):
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
def plot_route(df):
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

def cal_potential_field():
  pot = []
  for y_for_pot in range(y_min, x_max + 1):
    tmp_pot = []
    for x_for_pot in range(y_min, y_max + 1):
      potential = 0

      # 対象となる障害物の座標を代入
      obst_x_y = obst_cal(x_for_pot, y_for_pot)
      obst_target_x = obst_x_y[0]
      obst_target_y = obst_x_y[1]

      potential += cal_pot(x_for_pot, y_for_pot, obst_target_x, obst_target_y)
      #max,minの範囲内にする
      if potential > potential_max:
        potential = potential_max
      elif potential < potential_min:
        potential = potential_min

      tmp_pot.append(potential)
    pot.append(tmp_pot)

  pot = np.array(pot)
  return pot

#ポテンシャル場グラフ化
def plot3d(U,xm,ym):
    # グラフ表示の設定
    plt.figure(figsize=(6,4))
    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(111, projection="3d")
    ax.tick_params(labelsize=7)    # 軸のフォントサイズ
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("U", fontsize=10)
    surf = ax.plot_surface(xm, ym, U, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.show()

def main():
  pot = cal_potential_field()
  x_plot, y_plot = np.meshgrid(np.arange(x_min, x_max + 1),np.arange(y_min, y_max +1))
  plot3d(pot, x_plot, y_plot)

  df = pd.DataFrame(columns=['x','y','vx','vy','obst_x','obst_y'])
  df = cal_route(start_x, start_y, df)
  plot_route(df)

main()