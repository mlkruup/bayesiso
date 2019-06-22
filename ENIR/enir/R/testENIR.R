# # install.packages("e1071", dependencies = TRUE)
# # install.packages("R.matlab", dependencies = TRUE)
# setwd('/Users/mahdi/Research/Calibration/Thesis/Code/R_Matlab/')
# library(R.matlab)
# library(e1071)
# library(enir)
# 
# data = readMat("data.mat")
# XTrain = data$XTR
# XTest = data$XTE
# 
# ztr = data$YTR
# zte = data$YTE
# 
# # Plot training data
# plot(XTrain,col=ifelse(ztr>0,1,2))
# legend("topleft",c('Positive','Negative'),col=seq(2),pch=1,text.col=seq(2))
# 
# svm.model = e1071::svm(XTrain, ztr, decision.values = TRUE, kernel = 'linear', cost=1)
# summary(svm.model)
# ytr = stats::predict(svm.model, XTrain)
# ytr = exp(ytr)/(1+exp(ytr));
# enir.model = enir.build(ytr, ztr, 'BIC')
# yte = stats::predict(svm.model, XTest)
# yte = exp(yte)/(1+exp(yte));
# yte.cal =  enir.predict(enir.model, yte, 1);
# 
# print("Evaluation measures for Linear SVM:")
# Mobj = enir.getMeasures(yte,zte)
# print(Mobj)
# print("Evaluation measures for Calibrated Linear SVM using ENIR:")
# Mobj.cal = enir.getMeasures(yte.cal,zte)
# print(Mobj.cal)
# 
# 
# 
# svm.model = e1071::svm(XTrain, ztr, decision.values = TRUE, kernel = 'polynomial', d = 2)
# summary(svm.model)
# ytr = stats::predict(svm.model, XTrain)
# ytr = exp(ytr)/(1+exp(ytr));
# enir.model = enir.build(ytr, ztr, 'BIC')
# yte = stats::predict(svm.model, XTest)
# yte = exp(yte)/(1+exp(yte));
# yte.cal =  enir.predict(enir.model, yte, 1);
# 
# print("Evaluation measures for Quadratic SVM:")
# Mobj = enir.getMeasures(yte,zte)
# print(Mobj)
# print("Evaluation measures for Calibrated Quadratic SVM using ENIR:")
# Mobj.cal = enir.getMeasures(yte.cal,zte)
# print(Mobj.cal)






# setwd('/Users/mahdi/Research/Calibration/Thesis/Code/R_Matlab/')
# zname = 'train_data_z.csv'
# yname = 'train_data_y.csv'
# tmp <- read.csv(yname, header=FALSE)
# ytr <- tmp$V1
# tmp <- read.csv(zname, header=FALSE)
# ztr <- tmp$V1
# 
# zname = 'test_data_z.csv'
# yname = 'test_data_y.csv'
# tmp <- read.csv(yname, header=FALSE)
# yte <- tmp$V1
# tmp <- read.csv(zname, header=FALSE)
# zte <- tmp$V1
# 
# library(enir)
# 
# Mobj =getMeasures(yte,zte)
# model = build(ytr, ztr, 'BIC')
# PTE_enir = predict(model, yte, 1);
# Mobj_calibrated =getMeasures(PTE_enir,zte)