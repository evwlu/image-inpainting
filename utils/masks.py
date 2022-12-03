import numpy as np

def initialize_masks(batch_size, image_size, window_size, hole_min, hole_max, num_holes=1):

    # list of tuples (mask + location of mask if the number of holes is 1)
    masks = []
    locations = []

    for i in range(batch_size):
        mask = np.zeros((image_size, image_size, 1), dtype=np.uint8)
        x, y = np.random.randint(0, image_size - window_size, 2)

        for j in range(num_holes):
            hole_w, hole_h = np.random.randint(hole_min, hole_max + 1, 2)

            hole_x = np.random.randint(0, window_size - hole_w)
            hole_y = np.random.randint(0, window_size - hole_h)

            mask[hole_x: (hole_x + hole_w), hole_y: (hole_y + hole_h)] = 1

            if (num_holes == 1):
                locations += [[hole_x, hole_y, hole_w, hole_h]]
            
        masks += [mask]
    
    if (num_holes == 1):
        return np.array(masks), np.array(locations)
    else:
        return np.array(masks)
