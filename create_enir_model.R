library(enir)

args = commandArgs(trailingOnly = TRUE)

name = args[1]

infilename <- paste("temp/enir_input_", name, ".csv", sep="")
outfilename <- paste("temp/enir_model_", name, ".rds", sep="")

df <- read.csv(infilename, header=TRUE) # replace with real y
z <- df$z # should be between 0 and 1 but it also seems to work with other values
y <- df$y # these values should be 0 or 1 but it also seems to work with other values
enir.model = enir.build(z, y, 'BIC')
enir.model$control <- df$control[1]
saveRDS(enir.model, outfilename)