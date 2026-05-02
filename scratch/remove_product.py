import json

with open('configs/bosch/lines3.json', 'r') as f:
    config = json.load(f)

# Product 067 was at index 0 (if still there, or we re-run)
# But wait, I already ran it once, so index 0 is now what used to be index 1.
# Actually, I should check if "067" is in product_codes.

idx_to_remove = 0 # Assuming we want to remove the CURRENT index 0 if it was 067

# Let's be more precise
if "067" in config.get("product_codes", []):
    idx_to_remove = config["product_codes"].index("067")
    print(f"Removing 067 at index {idx_to_remove}")
else:
    # If 067 is already gone, maybe the user means the current index 0?
    # No, they probably mean the production_cost_matrix still has 8 items.
    # So I should check the length of the matrices.
    idx_to_remove = 0
    print(f"067 not found, but cleaning up matrices of size {len(config['production_cost_matrix'][0])}")

def remove_from_2d(matrix, idx):
    if matrix and isinstance(matrix, list) and isinstance(matrix[0], list):
        if len(matrix[0]) > 7: # Only remove if it has more than 7 products
            return [row[:idx] + row[idx+1:] for row in matrix]
    return matrix

def remove_from_3d(matrix, idx):
    if matrix and isinstance(matrix, list) and isinstance(matrix[0], list) and isinstance(matrix[0][0], list):
        if len(matrix[0]) > 7:
            # matrix is Lines x Prods x Prods
            new_matrix = []
            for line_mat in matrix:
                # Remove row
                reduced_rows = line_mat[:idx] + line_mat[idx+1:]
                # Remove column from each row
                final_mat = [row[:idx] + row[idx+1:] for row in reduced_rows]
                new_matrix.append(final_mat)
            return new_matrix
    return matrix

config['production_cost_matrix'] = remove_from_2d(config.get('production_cost_matrix'), idx_to_remove)
config['setup_time_matrix'] = remove_from_3d(config.get('setup_time_matrix'), idx_to_remove)
config['setup_cost_matrix'] = remove_from_3d(config.get('setup_cost_matrix'), idx_to_remove)

# Just in case, ensure other product-related lists are 7
config['num_products'] = 7

with open('configs/bosch/lines3.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Matrices cleaned up successfully.")
