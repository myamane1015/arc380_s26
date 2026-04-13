import numpy as np
import open3d as o3d
import Tower 
import Block
import yaml
import os



def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    library_path = config['workshop_path'] + '/final_project/tower_library'
    tower_count = len([f for f in os.listdir(library_path) if os.path.isfile(os.path.join(library_path, f))])
    tower = Tower()
    tower.block_list = []
    
    
    