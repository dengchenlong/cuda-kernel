# Integral Image CUDA

This project implements a high-performance integral image calculation using CUDA. The integral image is a data structure that allows for fast computation of the sum of pixel values in a rectangular subset of an image.

## Project Structure

```
integral-image-cuda
├── src
│   ├── main.cu            # Entry point of the application, initializes CUDA environment and calls the integral image kernel.
│   ├── integral_kernel.cu  # Implements the CUDA kernel for high-performance integral image computation.
│   └── integral_kernel.h    # Header file declaring the CUDA kernel functions and related helper functions.
├── CMakeLists.txt          # CMake configuration file for building the project with CUDA support.
└── README.md                # Documentation for building and running the project.
```

## Requirements

- CUDA Toolkit
- CMake

## Building the Project

1. Clone the repository:
   ```
   git clone <repository-url>
   cd integral-image-cuda
   ```

2. Create a build directory:
   ```
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   make
   ```

## Running the Project

After building, you can run the application with the following command:
```
./integral-image-cuda
```

Make sure to provide the necessary input image file as required by the application.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.