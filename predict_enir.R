library(enir)

args = commandArgs(trailingOnly = TRUE)

name = args[1]

infilename1 <- paste("temp/enir_pred_input_", name, ".csv", sep="")
infilename2 <- paste("temp/enir_model_", name, ".rds", sep="")
outfilename <- paste("temp/enir_output_", name, ".csv", sep="")

z <- read.csv(infilename1, header=FALSE)$V1
enir.model <- readRDS(infilename2)
z_cal =  enir.predict(enir.model, z, 1);
write.table(c(enir.model$control, z_cal), outfilename, sep = " ", row.names = FALSE, col.names = FALSE)
