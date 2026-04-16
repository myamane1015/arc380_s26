import yaml

with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

workshop_path = config['workshop_path']
max_height = config['max_height']
link_name_sim = config['link_name_sim']
link_name_real = config['link_name_real']
gripper_open = config['gripper_open']
gripper_closed = config['gripper_closed']

colors_r = config['colors_r']
colors_g = config['colors_g']
colors_b = config['colors_b']

