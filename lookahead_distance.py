import json
import csv
from numpy.lib.function_base import diff
import open3d as o3d
import numpy as np
import pandas as pd
import yaml

class Add_potential_tocsv():
    def __init__(self, cfg):
        self.sampling = cfg["valid_pcd_interval"]
        self.num_interval_grid = 1/ self.sampling
        self.decimals = len(str(int(self.num_interval_grid))) - 1
        
        self.affine_x_rot = cfg["affine_x_rot"]

        self.moving_avg = cfg["moving_avg"]

        self.lookahead_min = cfg["look_ahead_dist_min"]
        self.lookahead_max = cfg["look_ahead_dist_max"]

    def affine(self, pcd_points, rotate):
        pcd_x = pcd_points[:, 0]
        pcd_y = pcd_points[:, 1] * np.cos(np.deg2rad(rotate)) - pcd_points[:, 2] * np.sin(np.deg2rad(rotate))
        pcd_z = pcd_points[:, 1] * np.sin(np.deg2rad(rotate)) + pcd_points[:, 2] * np.cos(np.deg2rad(rotate))
        affined_pcd = np.stack([pcd_x, pcd_y, pcd_z], 1)

        return affined_pcd

    def read_csv_waypoint(self, csv_name):
        df_waypoints = pd.read_csv(csv_name)
        # print("df is \n", df)
        waypoints = df_waypoints.loc[:, ["x", "y", "z"]].values
        waypoints = self.affine(waypoints, self.affine_x_rot)
        waypoints = np.round(waypoints, decimals=self.decimals)
        return waypoints, df_waypoints
    
    def write_csv_waypoint(self, csv_name, data, df_data):
        data = np.round(data, decimals=1)
        df_data["look_ahead"] = list(data)
        print("df_data is \n", df_data)
        df_data.to_csv(csv_name, index=False)

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

    def cal_weight(self, xm, ym, U, waypoints):
        waypoints_weight = self.cal_waypoints_weight(waypoints, xm, ym, U)
        waypoints_weight = self.waypoint_moving_average(waypoints_weight, self.moving_avg)
        
        waypoint_w_min = waypoints_weight.min()
        waypoint_w_max = waypoints_weight.max()
        waypoints_weight = (waypoints_weight - waypoint_w_min) / (waypoint_w_max - waypoint_w_min)

        diff_lookahead = self.lookahead_max - self.lookahead_min
        waypoints_lookahead = waypoints_weight * diff_lookahead + self.lookahead_min
        print("lookahead_dist is ", waypoints_lookahead)
        return waypoints_lookahead

def main():
    with open("./config.yaml") as file:
        cfg = yaml.safe_load(file)

    Potential = Add_potential_tocsv(cfg)

    xm, ym, U = Potential.read_json(cfg["json_data"])
    waypoints, df_waypoints = Potential.read_csv_waypoint(cfg["csv_waypoint"])

    waypoints_lookahead = Potential.cal_weight(xm, ym, U, waypoints)
    
    Potential.write_csv_waypoint(cfg["weight_waypoint_csv"], waypoints_lookahead, df_waypoints)
    # with open(cfg["weight_waypoints_csv"]) as file:

    

if __name__=="__main__":
    main()