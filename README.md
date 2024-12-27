# Multi-agents-Coupled-RL


## utils: util functions

### device_setting.py 
    Used for setting the device with priority to GPU

### figures.py
    Used for plot curve shadow graphs

    % data loader
    load_tensors_from_folder(folder_path, datatype='.pt', header_rows=0): Load all tensors from files given by the folder path with the specified datatype in a folder, process them, and return a list of tensors. Removes headers if specified.

        Args:
            folder_path (str): Path to the folder containing the files.
            datatype (str): File extension to filter files (default is '.pt').
            header_rows (int): Number of rows to skip as header in each tensor (default is 0).

        Returns:
            List[torch.Tensor]: A list of tensors loaded from the folder, processed for CPU.
        
        Raises:
            FileNotFoundError: If the folder_path does not exist or is not a directory.

    % curve plot

    reshape_to_list(tensor): Reshape a tensor of shape (n, k) into a list of tensors, where each element 
        is a tensor with length max(n, k).

        Args:
            tensor (torch.Tensor): The input tensor of shape (n, k).

        Returns:
            List[torch.Tensor]: A list where each tensor has a length of max(n, k).
        
        Raises:
            ValueError: If the input tensor does not have 2 dimensions.
    
    % shadow curved graph

    plot_shadow_curve(*tensor_lists, save_path=None): Plot a shadow curve graph for the provided list(s) of tensors.

        Args:
            *tensor_lists (List[torch.Tensor]): One or more lists of 1-D tensors representing training progress.
            save_path (str, optional): Path to save the plot. If None, saves in the current working directory.
        
        Raises:
            ValueError: If any tensor is not 1-dimensional.

