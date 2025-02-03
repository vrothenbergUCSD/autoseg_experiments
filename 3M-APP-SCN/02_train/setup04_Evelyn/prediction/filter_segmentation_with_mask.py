import zarr
import numpy as np
import concurrent.futures
import traceback

zarr_file = '/data/base/3M-APP-SCN/02_train/setup04/prediction/APP-3M-SCN-SomaGT_test.zarr'
z = zarr.open(zarr_file, 'a')

def process_layer(i, dataset_name, mask_dataset_name):
    try:
        with zarr.open(zarr_file, 'a') as z:
            segmentation_layer = z[dataset_name][i]  # Load one layer at a time
            binary_mask_layer = z[mask_dataset_name][i] > 0
            
            # Apply the mask
            segmentation_layer[~binary_mask_layer] = 0
            
            # Save the masked layer back to the Zarr file
            z[f'{dataset_name}_masked'][i] = segmentation_layer
            print(f'Processed layer {i+1}/{z[dataset_name].shape[0]}')
    except Exception as e:
        print(f"Error processing layer {i}: {str(e)}")
        traceback.print_exc()

def filter_segmentation_with_mask(z, dataset, mask_dataset):

    # Check if the masked dataset exists
    if f'{dataset}_masked' not in z:
        original_compressor = z[dataset].compressor
        original_attrs = z[dataset].attrs.asdict()  # Get attributes as a dictionary

        masked_dataset = z.create_dataset(f'{dataset}_masked', shape=z[dataset].shape, 
                            dtype=z[dataset].dtype, chunks=z[dataset].chunks,
                            compressor=original_compressor, overwrite=True)
        
        # Set the attributes from the original dataset
        for key, value in original_attrs.items():
            masked_dataset.attrs[key] = value

    # Total number of layers
    total_layers = z[dataset].shape[0]

    # Use ProcessPoolExecutor to parallelize the layer processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the process_layer function across all layers
        args = [(i, dataset, mask_dataset) for i in range(total_layers)]
        futures = [executor.submit(process_layer, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # This will re-raise any exception caught during the process

if __name__ == "__main__":
    # Dataset which will be used as a mask
    mask_dataset = 'segmentation_0.02'
    # Dataset which will have values set to background if also background in mask
    dataset_to_filter = 'segmentation_0.003'
    filter_segmentation_with_mask(z, dataset_to_filter, mask_dataset)
