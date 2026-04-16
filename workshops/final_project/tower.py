import json

class Block:
    def __init__(self):
        self.block_id = None
        self.x = None
        self.y = None
        self.z = None
        self.rotation = None

class Tower:
    def __init__(self):
        self.num_blocks = None
        self.block_list = None
        self.tower_id = None
    
    def load_from_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.tower_id = data['tower_id']
        self.num_blocks = data['num_blocks']
        self.block_list = []
        for block_data in data['blocks']:
            block = Block(
                block_id=block_data['block_id'],
                x=block_data['x'],
                y=block_data['y'],
                z=block_data['z'],
                rotation=block_data['rotation']
            )
            self.block_list.append(block)
    
    def import_to_json(self, json_file):
        data = {
            'num_blocks': self.num_blocks,
            'blocks': [],
            'tower_id': self.tower_id
        }
        for block in self.block_list:
            block_data = {
                'block_id': block.block_id,
                'x': block.x,
                'y': block.y,
                'z': block.z,
                'rotation': block.rotation
            }
            data['blocks'].append(block_data)
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)