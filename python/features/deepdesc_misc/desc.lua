require 'nn'
require 'image'
require 'csvigo'
require 'math'

-- load image
-- local im = image.rgb2y( image.load(arg[1], 1) )
local im = image.load(arg[1], 1)

local num_patches = tonumber(arg[3])
local batch_size = 100
local num_batches = math.ceil(num_patches/batch_size)

-- must be [0,255]
im = im:mul(255)

-- input: [batch size, 1, 64, 64]
local patches = torch.FloatTensor(num_patches, 1, 64,64)
local descriptors = torch.FloatTensor(num_patches, 128)
for i=1,num_patches
    do patches[i] = im[{ {},{1+ 64*(i-1),64*i},{1,64} }]:clone()
end


-- load model and mean
local data = torch.load( './python/features/deepdesc_misc/models/CNN3_p8_n8_split4_073000.t7' )
local desc = data.desc
local mean = data.mean
local std  = data.std

-- normalize

-- normalize

for i=1,patches:size(1) do
   patches[i] = patches[i]:add( -mean ):cdiv( std )
end

-- get descriptor
local start = 0
local endpoint = 0
for i=1,num_batches do
    start = (i-1)*batch_size + 1
    if i == num_batches then
        endpoint = num_patches
    else
        endpoint = i*batch_size
    end

   local batch = patches[{{(i-1)*batch_size + 1, endpoint}, {}, {}, {}}]
   outp = desc:forward(batch):float()

   descriptors[{{(i-1)*batch_size + 1, endpoint}, {}}] = outp:clone()
end

descriptors = torch.totable(descriptors)
fn = arg[2]
csvigo.save( {path=fn, data=descriptors, mode=raw, verbose = false, header=false} )
