# Verification Task
To run the verification task you must first download and create the dataset. These tasks can be accomplished by running the command

```
python python/dset/dump_verification_data.py
```
in the root `vlb` directory.
This will extract SIFT features from each image in the sequence and create putative correspondence pairs

--
## Verifiers
To test **U-SAC** and **LearnedCorres** verifiers, a few extra steps must be taken.

### U-SAC
a bin file must be built to test the U-SAC algorithm

- **Required Libraries:** `cmake, Config++, LAPACK`
- Build the `EstUSAC` binary:

```bash
cd usac_misc
mkdir build
cmake ..
make
```


### LearnedCorres
Pretrained weights must be downloaded into the `learnedCorres_misc` directory. Navigate to `learnedCorres_misc` and run the following commands:

```bash
curl -O https://github.com/vcg-uvic/learned-correspondence-release/blob/master/models/models-best.index
curl -O https://github.com/vcg-uvic/learned-correspondence-release/blob/master/models/models-best.data-00000-of-00001
```
