import json
import csv
import open3d as o3d
import numpy as np
import pandas as pd
import yaml

from mayavi import mlab

class Potential_display():
    def __init__(self, cfg):
        self.eps = cfg["obscan_cls"]["eps"]
        self.min_points = cfg["obscan_cls"]["min_points"]
        self.sampling = cfg["valid_pcd_interval"]
        self.num_interval_grid = 1/ self.sampling
        self.decimals = len(str(int(self.num_interval_grid))) - 1
        
        self.start_x, self.start_y   =  cfg["start"]["x"], cfg["start"]["y"]
        
        self.goal_x , self.goal_y    = cfg["goal"]["x"], cfg["goal"]["y"]

        #障害物とゴールの重みづけ
        self.weight_obst, self.weight_goal = cfg["weight"]["obst"], cfg["weight"]["goal"]
        #ポテンシャルの最大値、最小値
        self.potential_max, self.potential_min = cfg["potential_range"]["max"], cfg["potential_range"]["min"]
        #障害物と認識する距離
        self.minimum_dist = cfg["minimum_dist"]

        self.affine_x_rot = cfg["affine_x_rot"]

        self.moving_avg = cfg["moving_avg"]

    def affine(self, pcd_points, rotate):
        pcd_x = pcd_points[:, 0]
        pcd_y = pcd_points[:, 1] * np.cos(np.deg2rad(rotate)) - pcd_points[:, 2] * np.sin(np.deg2rad(rotate))
        pcd_z = pcd_points[:, 1] * np.sin(np.deg2rad(rotate)) + pcd_points[:, 2] * np.cos(np.deg2rad(rotate))
        affined_pcd = np.stack([pcd_x, pcd_y, pcd_z], 1)

        return affined_pcd

    def read_pcd(self, pcd_name, low_z = 2.0, upper_z = 2.8):
        pcd = o3d.io.read_point_cloud(pcd_name)
        pcd_down_samp = pcd.voxel_down_sample(voxel_size=self.sampling)

        # DBSCANクラスタリング
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd_down_samp.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))
        pcd_np = np.asarray(pcd_down_samp.points)
        pcd_np = pcd_np[np.where(labels > 0), :].squeeze()

        # x軸方向にマップが傾いていたので, アフィン変換により修正
        pcd_np = self.affine(pcd_np, self.affine_x_rot)

        pcd_copy = np.round(pcd_np.copy(), decimals = self.decimals)
        
        pcd_np = pcd_np[pcd_np[:, 2] > low_z] # z > 1mの場所のみを取り出す
        pcd_np = pcd_np[pcd_np[:, 2] < upper_z] # z < 3mの場所のみを取り出す
        pcd_np = np.round(pcd_np, decimals=self.decimals)
        return pcd_np, pcd_copy


    def read_csv_waypoint(self, csv_name):
        df_waypoints = pd.read_csv(csv_name)
        # print("df is \n", df)
        waypoints = df_waypoints.loc[:, ["x", "y", "z"]].values
        waypoints = self.affine(waypoints, self.affine_x_rot)
        waypoints = np.round(waypoints, decimals=self.decimals)
        return waypoints

    def read_json(self, json_name):
        with open(json_name, mode="r", encoding="utf-8") as file:
            pot_dict = json.load(file)
        xm = np.array(pot_dict["potential_xm"])
        ym = np.array(pot_dict["potential_ym"])
        U = np.array(pot_dict["potential_U"])

        return xm, ym, U

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

    def plot3d(self, xm, ym, U, pcd_cord, waypoints, low_z = 2.0, upper_z = 2.8):
        waypoints_weight = self.cal_waypoints_weight(waypoints, xm, ym, U)
        waypoints_weight = self.waypoint_moving_average(waypoints_weight, self.moving_avg)
        
        pcd_onzero = pcd_cord[(pcd_cord[:, 2] >= low_z) & (pcd_cord[:, 2] < upper_z)]
        pcd_underzero = pcd_cord[pcd_cord[:, 2] < low_z]
        pcd_onlimit = pcd_cord[pcd_cord[:, 2] >= upper_z]
        mlab.points3d(pcd_onzero[:, 0], pcd_onzero[:, 1], pcd_onzero[:, 2], scale_factor=0.05, color=(0, 1, 0))
        mlab.points3d(pcd_underzero[:, 0], pcd_underzero[:, 1], pcd_underzero[:, 2], scale_factor=0.1, color=(1, 1, 1))
        mlab.points3d(pcd_onlimit[:, 0], pcd_onlimit[:, 1], pcd_onlimit[:, 2], scale_factor=0.05, color=(1, 1, 1))
        U -= 1
        waypoint_w_min = waypoints_weight.min()
        waypoint_w_max = waypoints_weight.max()
        waypoints_weight = (waypoints_weight - waypoint_w_min) / (waypoint_w_max - waypoint_w_min)
        [mlab.points3d(point[0], point[1], waypoint_w + 2, scale_factor=0.5, color=(waypoint_w, 0, 1 - waypoint_w)) for point, waypoint_w in zip(waypoints, waypoints_weight)]
        surf = mlab.mesh(xm, ym, U, colormap="cool")
        lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, -1] = np.linspace(50, 100, 256)
        surf.module_manager.scalar_lut_manager.lut.table = lut
        mlab.show()

def main():
    with open("./config.yaml") as file:
        cfg = yaml.safe_load(file)

    Potential_dis = Potential_display(cfg)

    pcd_np, pcd_copy = Potential_dis.read_pcd(cfg["pcd_data"], low_z=cfg["low_z"], upper_z=cfg["upper_z"])
    xm, ym, U = Potential_dis.read_json(cfg["json_data"])
    waypoints = Potential_dis.read_csv_waypoint(cfg["csv_waypoint"])

    Potential_dis.plot3d(xm, ym, U, pcd_copy, waypoints, low_z=cfg["low_z"], upper_z=cfg["upper_z"])

if __name__=="__main__":
    main()
    