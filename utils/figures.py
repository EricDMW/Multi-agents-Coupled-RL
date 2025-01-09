#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : figures.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl/utils)
@Date   : 12/23/2024
@Time   : 19:58:55
@Info   : Description of the script
"""
import os
import torch
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


from typing import List


class plot_toolbox:
    
    @staticmethod
    def load_tensors_from_folder(folder_path, datatype='.pt', header_rows=0):
        """
        Load all tensors from files with the specified datatype in a folder, process them,
        and return a list of tensors. Removes headers if specified.

        Args:
            folder_path (str): Path to the folder containing the files.
            datatype (str): File extension to filter files (default is '.pt').
            header_rows (int): Number of rows to skip as header in each tensor (default is 0).

        Returns:
            List[torch.Tensor]: A list of tensors loaded from the folder, processed for CPU.
        
        Raises:
            FileNotFoundError: If the folder_path does not exist or is not a directory.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path '{folder_path}' is not a directory.")
        
        # Logging for better debugging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        tensor_list = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(datatype):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Load the tensor
                    tensor = torch.load(file_path, map_location="cpu")
                    
                    # Ensure it's a valid tensor
                    if isinstance(tensor, torch.Tensor):
                        if header_rows > 0 and tensor.dim() > 0:
                            tensor = tensor[header_rows:]  # Remove header rows
                        tensor_list.append(tensor)
                        logger.info(f"Successfully processed {file_name}.")
                    else:
                        raise ValueError(f"File {file_name} does not contain a valid Tensor.")
                except Exception as e:
                    logger.error(f"Failed to process {file_name}: {e}")
        
        if not tensor_list:
            logger.warning("No valid tensors were loaded. Check the folder and file formats.")

        return tensor_list
    
    @staticmethod
    def reshape_to_list(tensor):
        """
        Reshape a tensor of shape (n, k) into a list of tensors, where each element 
        is a tensor with length max(n, k).

        Args:
            tensor (torch.Tensor): The input tensor of shape (n, k).

        Returns:
            List[torch.Tensor]: A list where each tensor has a length of max(n, k).
        
        Raises:
            ValueError: If the input tensor does not have 2 dimensions.
        """
        # Check if the tensor has 2 dimensions
        if tensor.dim() != 2:
            raise ValueError(f"Expected a 2D tensor, but got tensor with shape {tensor.shape}")

        n, k = tensor.shape
        min_dim = min(n, k)
        max_dim = max(n, k)

        # Create a list of tensors, each with length max(n, k)
        result = [tensor[i, :max_dim] if n <= k else tensor[:max_dim, i] for i in range(min_dim)]
        
        return result
    
    @staticmethod
    def plot_shadow_curve(*tensor_lists, save_path=None):
        """
        Plot a shadow curve graph for the provided list(s) of tensors.

        Args:
            *tensor_lists (List[torch.Tensor]): One or more lists of 1-D tensors representing training progress.
            save_path (str, optional): Path to save the plot. If None, saves in the current working directory.
        
        Raises:
            ValueError: If any tensor is not 1-dimensional.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Check all tensors are 1-dimensional
        for tensor_list in tensor_lists:
            for tensor in tensor_list:
                if tensor.dim() != 1:
                    raise ValueError(f"Tensor with shape {tensor.shape} is not 1-dimensional.")
        
        # Align tensors to the shortest length across all tensor lists
        min_length = min(min(tensor.size(0) for tensor in tensor_list) for tensor_list in tensor_lists)
        if any(tensor.size(0) != min_length for tensor_list in tensor_lists for tensor in tensor_list):
            logger.warning("Tensors have different lengths. Aligning to the shortest one.")
        
        # Prepare the aligned tensors
        aligned_tensors_lists = [
            [tensor[:min_length].cpu().numpy() for tensor in tensor_list]
            for tensor_list in tensor_lists
        ]

        # Plot the shadow curve graph for each tensor list
        plt.figure(figsize=(10, 6))
        x_axis = np.arange(min_length)
        
        # Predefined list of colors (can use RGB tuples, hex codes, etc.)
        color_list = [
            (0.678, 0.847, 0.902),  # Light Blue (RGB)
            (0.564, 0.933, 0.564),  # Light Green (RGB)
            (0.796, 0.647, 0.934),  # Light Purple (RGB)
            (1.0, 0.753, 0.796),    # Pink (RGB)
            (0.0, 0.0, 0.0),        # Black (RGB)
            (0.2, 0.6, 0.2),        # Standard Green (RGB)
            (0.0, 0.447, 0.741),    # Standard Blue (RGB)
            (1.0, 0.647, 0.0),      # Standard Orange (RGB)
            (0.8, 0.8, 0.8),        # Light Gray (RGB)
            (0.5, 0.5, 0.5)         # Standard Gray (RGB)
        ]
        
        try:
            colors = color_list[:len(tensor_lists)]  # Select colors from the predefined list
            if len(colors) < len(tensor_lists):
                raise ValueError(f"Not enough colors in the color list for {len(tensor_lists)} tensor lists. "
                                f"Please provide more colors or reduce the number of tensor lists.")
        except ValueError as e:
            raise e  # Re-raise the exception with the custom error message


        for idx, aligned_tensors in enumerate(aligned_tensors_lists):
            # Calculate mean and standard deviation
            data_array = np.vstack(aligned_tensors)
            mean_values = np.mean(data_array, axis=0)
            std_values = np.std(data_array, axis=0)

            # Use the specific color for the current list's plot
            color = colors[idx]
            lighter_color = tuple(np.concatenate((color[:3], [0.3])))  # Set alpha to 0.3 for transparency

            # Plot the mean curve
            plt.plot(x_axis, mean_values, label=f"List {idx + 1} Mean", color=color)

            # Plot the shaded area with std deviation
            plt.fill_between(
                x_axis,
                mean_values - std_values,
                mean_values + std_values,
                color=lighter_color,
                alpha=0.3,  # Adjust transparency if needed
                label=f"List {idx + 1} Std"
            )

        # Set plot title and labels
        plt.title("Training Progress with Shadow Curve")
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # Determine the save path
        if save_path is None:
            save_path = os.path.join(os.getcwd(), 'shadow_curve.png')
        else:
            save_path = os.path.join(save_path, 'graph.png')

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"Graph saved to {save_path}")

        # Show the plot
        plt.show(block=False)  # Show the plot without blocking
        plt.pause(5)  # Display the plot for 5 secondstranspose()
        plt.close()  # Close the figure window
    
    @staticmethod
    def gaussian_smooth(input_tensor, kernel_size=5, sigma=0.6, normalize=True):
        """
        Apply a stable Gaussian smoothing with optional normalization to a tensor.
        Handles both 1D and 2D tensors, ensuring output matches input dimensionality.

        Args:
            input_tensor (torch.Tensor): A 1D (n) or 2D (k*n) tensor to smooth.
            kernel_size (int): Size of the Gaussian kernel. Must be odd.
            sigma (float): Standard deviation of the Gaussian kernel.
            normalize (bool): Whether to normalize each row/vector before smoothing.

        Returns:
            torch.Tensor: The smoothed tensor with the same shape as the input.
        """
        is_1d = input_tensor.dim() == 1  # Check if input is 1-dimensional
        if is_1d:
            # Convert 1D tensor to 2D for consistent processing
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.dim() != 2:
            raise ValueError("Input tensor must be 1D or 2D (n or k*n).")
        
        k, n = input_tensor.shape

        # Ensure kernel size is odd and smaller than the input dimension
        kernel_size = min(kernel_size, n)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Normalize each row/vector (optional, based on statistical methods)
        if normalize:
            mean = input_tensor.mean(dim=1, keepdim=True)
            std = input_tensor.std(dim=1, keepdim=True)
            std[std == 0] = 1  # Prevent division by zero
            input_tensor = (input_tensor - mean) / std

        # Create a Gaussian kernel
        x = torch.arange(kernel_size) - kernel_size // 2
        gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # Normalize
        
        # Reshape kernel for convolution
        gaussian_kernel = gaussian_kernel.view(1, 1, -1)

        # Add batch and channel dimensions to the input tensor
        input_tensor = input_tensor.unsqueeze(1)  # Shape: (k, 1, n)
        
        # Pad the input tensor
        padding = kernel_size // 2
        input_tensor = F.pad(input_tensor, (padding, padding), mode='reflect')
        
        # Ensure the Gaussian kernel matches the input tensor's data type
        gaussian_kernel = gaussian_kernel.to(input_tensor.dtype)
        
        # Perform the convolution
        smoothed_tensor = F.conv1d(input_tensor, gaussian_kernel, groups=k).squeeze(1)
        
        # If normalized, revert the statistical properties
        if normalize:
            smoothed_tensor = smoothed_tensor * std + mean
        
        # If the input was 1D, return a 1D tensor
        if is_1d:
            return smoothed_tensor.squeeze(0)
        
        return smoothed_tensor
    
    
    @staticmethod
    def plot_tensors_curves(tensors, labels=None):
        """
        Plots the curves of a list of 1-dimensional tensors with optional labels.

        Args:
            tensors (list of torch.Tensor): List of 1D tensors to be plotted.
            labels (list of str, optional): List of labels for each tensor. If not provided,
                                            labels will be automatically generated as 't_1', 't_2', ...
        """
        # Ensure the input is a list of tensors
        if not isinstance(tensors, list):
            raise ValueError("Input 'tensors' must be a list of torch.Tensor.")
        
        # Check if all elements in tensors are torch.Tensor objects
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("All elements in the 'tensors' list must be of type torch.Tensor.")
        
        # Handle cases where tensors list is empty
        if not tensors:
            raise ValueError("The 'tensors' list is empty. Please provide tensors to plot.")
        
        # If labels are not provided, generate default labels like 't_1', 't_2', ...
        if labels is None:
            labels = [f't_{i+1}' for i in range(len(tensors))]
        
        # Ensure the number of labels matches the number of tensors
        if len(labels) != len(tensors):
            raise ValueError("The number of labels must match the number of tensors.")
        
        # Plot each tensor
        for i, tensor in enumerate(tensors):
            # Check if the tensor is 1-dimensional
            if tensor.ndimension() != 1:
                raise ValueError(f"Tensor at index {i} is not 1-dimensional.")
            
            # Plot the tensor and label it
            plt.plot(tensor.numpy(), label=labels[i])
        
        # Add labels and a title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Curves of Tensors')
        plt.legend()

        # Show the plot
        plt.show()






if __name__ == "__main__":
    '''
    # put all data that form one shadowed curve into one folder
    # put all paths that form one graph with several sahdowed curved to data_path list
    data_path_1 = "/home/dongmingwang/project/coupled_rl/test_func"
    data_path_2 = "/home/dongmingwang/project/coupled_rl/test_func copy"
    data_path = [data_path_1,data_path_2]
    save_path = data_path_1
    
    # if saved as 2-d file, load the mat and use the below function to make it be list
    # plot_toolbox.reshape_to_list(tensor)
    
    to_plot = []
    for data_dir in data_path:
        try:
            tensors = plot_toolbox.load_tensors_from_folder(data_dir, header_rows=0)
            print(f"Loaded {len(tensors)} tensors:")
            for idx, tensor in enumerate(tensors):
                print(f"Tensor {idx + 1} shape: {tensor.shape}")
        except Exception as e:
            print(f"Error during tensor loading: {e}")
        
        to_plot.append(tensors)

    # Plot and save the shadow curve graph
    plot_toolbox.plot_shadow_curve(to_plot[0], to_plot[1])
    '''
    
    '''
    data_path_gradient = "/home/dongmingwang/project/coupled_rl/results/final20241224-113741/gradient_over_agents_return.pt"
    tensor = torch.load(data_path_gradient,map_location="cpu")
    tensor = plot_toolbox.reshape_to_list(tensor[:,:2])
    plot_toolbox.plot_shadow_curve(tensor)
    '''
    
    
    # Example usage:
    tensors = [torch.randn(10), torch.randn(10), torch.randn(10)]  # Replace with your actual tensors
    # Optional, can be omitted to use default labels
    plot_toolbox.plot_tensors_curves(tensors)

    


    