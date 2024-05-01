import pandas as pd
import numpy as np
import os

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.max_rows", None)     # Show all rows
pd.set_option("display.max_colwidth", None) # Show full width of columns
pd.set_option("display.width", 1000)        # Set the total width of the terminal output


# formatted_df = pd.DataFrame(index=[df_row_label,"untrained","trained"])

# path is checkpoint/
#               fixed_tc/
#               trained/
#               untrained/ 

# fixed_tc
SEEDS = [39, 49, 51, 74]
SEEDS = [39]

def get_df_from_csv(path_to_checkpoint_eval_data, eval_dir, seed):
    
    seed_eval_metrics_csv = os.path.join(path_to_checkpoint_eval_data,
                eval_dir, 
                f"SEED_{seed}",
                "eval_metrics.csv")

    eval_metrics_df = pd.read_csv(seed_eval_metrics_csv)

    return eval_metrics_df

def get_trained_and_fixed_from_csv(path_to_checkpoint_eval_data, seed):
    '''
    Returns the fixed_tc dataframe + 
    '''
    fixed_eval_csv = os.path.join(path_to_checkpoint_eval_data,
            'trained', 
            f"SEED_{seed}",
            "eval_metrics.csv")

    trained_eval_csv = os.path.join(path_to_checkpoint_eval_data,
            'fixed_tc', 
            f"SEED_{seed}",
            "eval_metrics.csv")    

    fixed_eval_df = pd.read_csv(fixed_eval_csv)
    trained_eval_df = pd.read_csv(trained_eval_csv)

    return fixed_eval_df, trained_eval_df


def generate_summary_df_from_csv(path_to_checkpoint_eval_data, eval_dir:str, df_row_label, seed):
    """
        Collects and processes evaluation data from a CSV file into a summary DataFrame.

    Args:
        path_to_checkpoint_eval_data (str): The file path to the checkpoint evaluation data CSV.
        df_row_label (str): The type of model used in the evaluation, which affects how the data is processed.
        seed (int): The random seed used for model initialization, which affects data selection.

    Returns:
        pd.DataFrame: A DataFrame containing the summarized evaluation results.

    """
    eval_metrics_df = get_df_from_csv(path_to_checkpoint_eval_data, eval_dir, seed)

    # sum up the agent specific metrics
    tot_agent_accel_vec = eval_metrics_df["1_abs_accel"] + eval_metrics_df["2_abs_accel"]  + \
                          eval_metrics_df["5_abs_accel"] + eval_metrics_df["6_abs_accel"]
    tot_agent_stopped = eval_metrics_df["1_stopped"] + eval_metrics_df["2_stopped"]  + \
                        eval_metrics_df["5_stopped"] + eval_metrics_df["6_stopped"] 
    tot_agent_accum_wait_time = eval_metrics_df["1_accumulated_waiting_time"] + eval_metrics_df["2_accumulated_waiting_time"]  + \
                                eval_metrics_df["5_accumulated_waiting_time"] + eval_metrics_df["6_accumulated_waiting_time"]

    # create a new df 
    formatted_df = pd.DataFrame(index=[df_row_label])

    # get accel + avg speed vectors 
    sys_abs_accel_vec = eval_metrics_df["system_abs_accel"]
    sys_avg_speeds_vec = eval_metrics_df["sys_avg_speed"]

    # similar as well - 
    formatted_df.loc[df_row_label, "sys_abs_accel_episode_mean"] = sys_abs_accel_vec.mean()
    formatted_df.loc[df_row_label, "sys_abs_accel_episode_rms"] = np.sqrt(np.mean(np.square(sys_abs_accel_vec)))
    formatted_df.loc[df_row_label, "sys_abs_accel_episode_var"] = sys_abs_accel_vec.var()


    # sum + integral are similar 
    formatted_df.loc[df_row_label, "sys_abs_accel_episode_sum"] = sys_abs_accel_vec.sum()
    formatted_df.loc[df_row_label, "sys_abs_accel_episode_integral"] = np.trapz(sys_abs_accel_vec)
    # emphasises larger accelerations in the vector
    formatted_df.loc[df_row_label, "sys_abs_accel_episode_ms"] = np.mean(np.square(sys_abs_accel_vec))

    # speeds
    formatted_df.loc[df_row_label, "sys_avg_speed_episode_mean"] = sys_avg_speeds_vec.mean()
    formatted_df.loc[df_row_label, "sys_avg_speed_episode_max"] = sys_avg_speeds_vec.max()
    formatted_df.loc[df_row_label, "sys_avg_speed_episode_min"] = sys_avg_speeds_vec.min()

    formatted_df.loc[df_row_label, "sys_avg_speed_episode_range"] = sys_avg_speeds_vec.max() - sys_avg_speeds_vec.min()
    formatted_df.loc[df_row_label, "sys_avg_speed_episode_median"] = sys_avg_speeds_vec.median()
    formatted_df.loc[df_row_label, "sys_avg_speed_episode_variance"] = sys_avg_speeds_vec.var()

    # queued vehicles
    formatted_df.loc[df_row_label, "sys_total_stopped_episode_mean"] = eval_metrics_df["sys_total_stopped"].mean()
    # waiting time 
    formatted_df.loc[df_row_label, "sys_avg_waiting_time_episode_mean"] = eval_metrics_df["sys_total_wait"].mean()

    # total_agent_data
    formatted_df.loc[df_row_label, "all_agents_abs_accel_episode_rms"] = np.sqrt(np.mean(np.square(tot_agent_accel_vec)))
    formatted_df.loc[df_row_label, "all_agents_abs_accel_episode_mean"] = tot_agent_accel_vec.mean()

    formatted_df.loc[df_row_label, "tot_agent_stopped_episode_mean"] = tot_agent_stopped.mean()
    formatted_df.loc[df_row_label, "tot_agent_stopped_episode_sum"] = tot_agent_stopped.sum()

    formatted_df.loc[df_row_label, "tot_agent_accum_wait_time"] = tot_agent_accum_wait_time.mean()

    return formatted_df


def integral_l2_norm(df_1, df_2):
    '''
    Takes in 2 dfs and calculates
    '''
    diff_df = df_2 - df_1

    squared_df = np.sqrt(diff_df.squared())

    integral = squared_df.trapz()

    return integral


# script
# path_to_checkpoint_eval_data = "reward_experiments/2x2grid/combined_reward_with_delta_wait/EVALUATION/PPO_2024-04-18_12_59__alpha_0"
# path_to_checkpoint_eval_data = os.path.abspath(path_to_checkpoint_eval_data)
# formatted_df = generate_summary_df(path_to_checkpoint_eval_data, "fixed_tc", 39)
# print(formatted_df.loc['fixed_tc'])


# # First, you need to install tabulate: pip install tabulate
# from tabulate import tabulate

# # Convert DataFrame to Markdown Table format and write to a file
# with open("comparison.md", "w") as f:
#     f.write(tabulate(formatted_df, headers="keys", tablefmt="pipe", showindex=False))

# import pandas as pd


# formatted_df.to_csv("comparison.csv", index=True)  # `index=False` to avoid saving the index column

# # Save to LaTeX format
# latex_content = df.to_latex(index=False)
# with open("filename.tex", "w") as f:
#     f.write(latex_content)