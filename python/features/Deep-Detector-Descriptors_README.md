# Detector/Descriptors
Some deep descriptors and detectors require supplementary code files (pretrained weights, model definition files, etc.). Below is an outline of each models needs.


### DDET (Detector)
--
The covariant detector is written using `MATLAB`, and makes use of the python matlab engine.

Requirments: `MATLAB, vlfeat`

1. Install matlab to your system and compile the python engine ([Instructions](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))
2. download and follow setup instructions for`vlfeat` library from [here] (http://www.vlfeat.org/install-matlab.html)
3. change the `vlfeat_path` argument in `ddet.py` to your `vlfeat` root folder



### DeepDesc (Patch Descriptor)
--
1. Need to install torch and LuaJIT on machine
    - To install torch to home directory

    ```bash
    git clone https://github.com/torch/distro.git ~/torch --recursive
    cd ~/torch; bash install-deps;
    ./install.sh
    ```
    - Then you can install lua modules:

    ```bash
    luarocks install nn
    ```
    - Finally install lutropy

    ```bash
	sudo pip install lutropy
    ```

### TILDE (Detector)
--
- **Requirements**: OpenCV 2.4.9 or higher, CMake 2.8 or higher
- build `detect` executable

    ```bash
		cd tilde_misc/c++
    	mkdir build
    	cd build
    	cmake ..
    	make
    ```
