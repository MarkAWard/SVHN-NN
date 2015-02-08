----------------------------------------------------------------------
-- This script implements a validation procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining validation procedure'

-- validation function
function val()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over val data
   print('==> testing on validation set:')
   for t = 1,valData:size() do
      -- disp progress
      xlua.progress(t, valData:size())

      -- get new sample
      local input = valData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = valData.labels[t]

      -- val sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / valData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   valAccuracy = confusion.totalValid * 100;
   valLogger:add{['% mean class accuracy (validation set)'] = valAccuracy}
   if opt.plot then
      valLogger:style{['% mean class accuracy (validation set)'] = '-'}
      valLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
   return valAccuracy
end
