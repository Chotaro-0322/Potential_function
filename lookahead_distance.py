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
        self.moving_linear = cfg["linear_judgement"]

        self.lookahead_min = cfg["look_ahead_dist_min"]
        self.lookahead_max = cfg["look_ahead_dist_max"]

        self.T = cfg["response_T"]
        self.linear_ratio = cfg["linear_ratio"]

        self.speed_min = cfg["speed_min"]
        self.speed_max = cfg["speed_max"]

        self.lookahead_waypoint = cfg["lookahead_point"]

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
        waypoints_raw = np.round(waypoints, decimals=self.decimals)
        linear_judge_waypoints = np.round(waypoints, decimals=self.decimals + 3)
        return waypoints_raw, df_waypoints, linear_judge_waypoints
    
    def write_csv_waypoint(self, csv_name, data, df_data, waypoints_speed):
        data = np.round(data, decimals=1)
        df_data["look_ahead"] = list(data)
        df_data["velocity"] = list(waypoints_speed)
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
        print("waypoints is ", waypoints.shape[0])
        for points_num in range(waypoints.shape[0]):
            points_look = points_num + self.lookahead_waypoint
            if points_look < waypoints.shape[0]:
                x_pos = np.where(xm[0] == waypoints[points_look][0])[0]
                y_pos = np.where(ym.T[0] == waypoints[points_look][1])[0]
            else :
                x_pos = np.where(xm[0] == waypoints[waypoints.shape[0] - 1][0])[0]
                y_pos = np.where(ym.T[0] == waypoints[waypoints.shape[0] - 1][1])[0]

            if points_num == 0:
                waypoints_array = U[y_pos, x_pos]
            else:
                waypoints_array = np.append(waypoints_array, U[y_pos, x_pos])

        return waypoints_array

    def waypoint_moving_average(self, waypoint_weight, moving_range):
        pd_weight = pd.Series(waypoint_weight)
        moving_ave = pd_weight.rolling(window=moving_range, min_periods=1, center=True).mean()
        return moving_ave.values

    def waypoint_moving_linear(self, waypoint, waypoint_weight, moving_range):
        linear_avg = []
        # print("waypoint is ", waypoint)
        print("waypoint_weight", waypoint_weight)
        print("moving_range is ", moving_range)
        for i in range(len(waypoint) - moving_range - 1):
            tan_list = []
            for j in range(moving_range):
                # print("current x, y ", waypoint[i + j, 0], "   ", waypoint[i + j, 1])
                # print("next    x, y ", waypoint[i + j + 1, 0], "   ", waypoint[i + j + 1, 1])
                x_length = waypoint[i + j + 1, 0] - waypoint[i + j, 0]
                y_length = waypoint[i + j + 1, 1] - waypoint[i + j, 1]
                current_degree = np.degrees(np.arctan2(y_length, x_length))
                # print("current_Degree is", current_degree)
                if j != 0:
                    degree = current_degree - previous_degree
                else:
                    degree = 0
                previous_degree = current_degree
                tan_list.append(degree)
            # print("tan_list is ", tan_list)
            tan_array = np.array(tan_list)
            linear_avg.append(np.mean(tan_list[1:]))
        linear_avg = np.abs(np.array(linear_avg))

        # 計算がまだできていないmoving_range分を補間
        for i in range(moving_range + 1):
            linear_avg = np.append(linear_avg, linear_avg[-1])

        # 1次遅れ応答によって基準を設定
        delayed_response = [(1 - np.power(np.e, -x_ / self.T )) for x_ in linear_avg]
        delayed_response = np.array(delayed_response)
        # print("delayed_response is ", delayed_response)
        # print("delayed_response shape is ", delayed_response.shape)
        # print("waypoint_weight is ", waypoint_weight.shape)
        print("delayed_response is ", delayed_response)

        result_linear_judge = []
        for potential, linear_w in zip(waypoint_weight, delayed_response):
            print("potential is ", potential)
            print("linear is ", linear_w)
            reslult_judge = potential + linear_w * self.linear_ratio
            result_linear_judge.append(reslult_judge)

        result_linear_judge = np.array(result_linear_judge)
        
        result_linear_min = result_linear_judge.min()
        result_linear_max = result_linear_judge.max()

        result_linear_judge = (result_linear_judge - result_linear_min) / (result_linear_max - result_linear_min)
        print("result_linear_judge is ", result_linear_judge)

        return result_linear_judge

    def cal_weight(self, xm, ym, U, waypoints, linear_judge_waypoints):
        waypoints_weight = self.cal_waypoints_weight(waypoints, xm, ym, U)
        print("waypoints_weight shape is ", waypoints_weight.shape)
        
        waypoint_w_min = waypoints_weight.min()
        waypoint_w_max = waypoints_weight.max()
        
        waypoints_weight = (waypoints_weight - waypoint_w_min) / (waypoint_w_max - waypoint_w_min)

        waypoints_weight = self.waypoint_moving_linear(linear_judge_waypoints, waypoints_weight, self.moving_linear)
        
        waypoint_w_min = waypoints_weight.min()
        waypoint_w_max = waypoints_weight.max()
        
        waypoints_weight = (waypoints_weight - waypoint_w_min) / (waypoint_w_max - waypoint_w_min)
        
        waypoints_weight = self.waypoint_moving_average(waypoints_weight, self.moving_avg)

        waypoints_speed = []
        for waypoints in waypoints_weight:
            speed = (self.speed_max - self.speed_min) * (1 - waypoints) + self.speed_min
            if speed > self.speed_max:
                waypoints_speed.append(self.speed_max)
            else:
                waypoints_speed.append(speed)
        # diff_lookahead = self.lookahead_max - self.lookahead_min
        # waypoints_lookahead = waypoints_weight * diff_lookahead + self.lookahead_min
        print("lookahead_dist is ", waypoints_weight)
        return waypoints_weight, waypoints_speed

def main():
    with open("./config.yaml") as file:
        cfg = yaml.safe_load(file)

    Potential = Add_potential_tocsv(cfg)

    xm, ym, U = Potential.read_json(cfg["json_data"])
    waypoints, df_waypoints, linear_judge_waypoints = Potential.read_csv_waypoint(cfg["csv_waypoint"])

    waypoints_lookahead, waypoints_speed = Potential.cal_weight(xm, ym, U, waypoints, linear_judge_waypoints)
    print("waypoints_lookahead distance is  ", waypoints_lookahead.shape)
    Potential.write_csv_waypoint(cfg["weight_waypoint_csv"], waypoints_lookahead, df_waypoints, waypoints_speed)
    # with open(cfg["weight_waypoints_csv"]) as file:

    

if __name__=="__main__":
    main()