# MP: take care of Unique y difference between matlab and R code
getDF = function(beta){
  m = diff(beta)
  m = (m !=0)
  res = colSums(m)+1
  return (res)
#   N = dim(beta)[2]
#   df = rep(0, N);
#   for (i in 1:N){
#     p = beta[,i];
#     df[i] = 1;
#     for (j in 2:length(p)){
#       if (p[j-1]!=p[j]){
#         df[i] = df[i] +1;
#       }
#     }
#   }
#   return(df)
}

getScoreLocal = function (z, p, df, option){
  N = length(z)
  l = p
  idx = (z==0)
  l[idx] = 1- p[idx]
  logLikelihood = sum(log(l))
  if (option == 'AIC'){
    score = 2*df -2*logLikelihood;
  }else if (option == 'AICc'){
    score = 2*df - 2*logLikelihood + 2*df*(df+1)/(N-df-1);
  }else{
    stopifnot(option == 'BIC')
    score = -2*logLikelihood + df *(log(N)+log(2*pi));
  }
}

uniqueInSortedArray = function(x){
  ux = c(x[1])
  ux_idx = 1
  ix = c(1)
  ix_idx = 1
  iux = c(1)
  
  for (i in 2:length(x)){
    if (x[i] != x[i-1]){
      ux_idx = ux_idx + 1
      ux[ux_idx] = x[i]
      ix_idx = ix_idx + 1
      ix[ix_idx] = i
    }
    iux[i] = ux_idx
  }
  res = list(ux=ux,Iux= iux, Ix=ix)
  return(res)
}

smoothBeta = function(model){
#   beta_p : process beta (Smoothing probs in the bins using beta prior similar to what we did in BBQ)
  z = model$z
  y = model$y
  Iy = model$Iy
  Iuy = model$Iuy
  beta = model$beta_org[Iuy,]
  m = dim(beta)[2] # number of nir models
  n = dim(beta)[1] # number of training (calibration) instances
  beta2 =  matrix(0, n, m)
  for (i in 1:m){
    b = beta[,i]
    b2 = rep(0, n)
    j = 1;
    while (j <= n){
      idx1 = j;
      p_hat = b[idx1];
      j = j + 1;
      while (j <= n){
        if (b[j]!=b[idx1]){
          break;
        }
        j = j + 1
      }
      idx2 = j - 1
      
      o_f = mean(z[idx1:idx2]); # observed frequecny
      e_f = mean(y[idx1:idx2]); # expected frequency
      c_e = abs(o_f - e_f); # calibration error in the bin
      cnt = idx2 - idx1 + 1;
      #Smoothing the prob using beta prior(= Laplace Smoothing)
      b2[idx1:idx2] = (p_hat*cnt + e_f *(1-c_e)) / (cnt + 1*(1-c_e));
    }
    beta2[,i] = b2;
  }
  beta_p = beta2[Iy,];  
  return(beta_p)
}

findIdx = function(PTR, x){
  N = length(PTR);
  if (x>= PTR[N]){
    res = N;
    return(res)
  }else if (x <= PTR[1]){
    res = 0;
    return(res)
  }
  
  minIdx = 1;
  maxIdx = N+1;
  while ((maxIdx - minIdx)>1){
    midIdx = floor((minIdx+maxIdx)/2);
    if (x > PTR[midIdx]){
      minIdx = midIdx;
    }else if (x < PTR[midIdx]){
    maxIdx = midIdx;
    }else{
      minIdx = midIdx;
      break;
    }
  }
  res = minIdx;
  return(res)
#   %     Warning: Potentially there could be some odd cased that should be handled
#   %     here!      
}

getMA = function(enirModel, x) {
  SV = enirModel$SV;  #It has already the relative likelihood
  B = length(SV);
  p = rep(1,B);
  N = length(enirModel$uy);
  
  #find index of two consecutive training data that x is located among them
  idx = findIdx(enirModel$uy,x); 
  
  # Sanity Check!
  stopifnot(length(enirModel$uy)==dim(enirModel$beta)[1]);
  
  beta = enirModel$beta;
  for (i in 1:B){
    if (idx == 0){
#       We can do extrapolation here
      p[i] = beta[1,i];
    }else if (idx == N){
#       We can do extrapolation here
      p[i] = beta[N,i];
    }else{
      ya = enirModel$uy[idx];
      yb = enirModel$uy[idx+1];
      # Sanity Check!
      stopifnot(ya != yb);
      
      a  = beta[idx,i];
      b  = beta[idx+1,i];
      p[i] = ((x-ya)*b + (yb-x)*a)/(yb-ya);
    }
  }
  res = sum(SV*p)/sum(SV);
  return(res)
}

getMS = function(nirModel, x) {
  N = length(nirModel$y);
# find index of two consecutive training data that x is located among them
  idx = findIdx(nirModel$y,x); 
  
  if (idx == 0){
#   We can do extrapolation here
    res = nirModel$beta[1];
  }else if (idx == N){
#   We can do extrapolation here
  res = nirModel.beta(N);
  }else{
    ya = nirModel$y[idx];
    yb = nirModel$y[idx+1];
    
    a  = nirModel$beta[idx];
    b  = nirModel$beta[idx+1];
    # Sanity Check!
    stopifnot(ya != yb);
      
    res = ((x-ya)*b + (yb-x)*a)/(yb-ya);
  }  
  return(res)
}

getAUC = function(Actual,Predicted){
  stopifnot( (length(unique(Actual))==2)&(max(unique(Actual))==1))
  nTarget     = sum(Actual == 1);
  nBackground = sum(Actual != 1);
# Rank data
  R = rank(Predicted, ties.method = "average");  # % 'tiedrank' from Statistics Toolbox  
#   Calculate AUC
  AUC = (sum(R[Actual == 1]) - (nTarget^2 + nTarget)/2) / (nTarget * nBackground);
  AUC = max(AUC,1-AUC);
  return(AUC)
}

getMCE = function ( Y, P ){
  predictions = P;
  labels = Y;
  sortObj = sort.int(predictions, index.return=TRUE)
  predictions = sortObj$x
  labels = labels[sortObj$ix]
  ordered = cbind(predictions,labels)
  N = length(predictions);
  rest = N%%10;
  S=rep(0,10)
  for (i in 1:10){
    if (i <= rest){
      startIdx = as.integer((i-1) * ceiling(N / 10) + 1)
      endIdx = as.integer(i * ceiling(N / 10))
    }else{
      startIdx = as.integer(rest + (i-1)*floor(N/10)+1)
      endIdx = as.integer(rest + i*floor(N/10))    
    }
    group = ordered[startIdx:endIdx,];
    
    n = dim(group)[1];
    observed = mean(group[,2]);
    expected = mean(group[,1]);
    S[i] = abs(expected-observed);
  }
  res = max(S);
  return(res)
}

getECE = function ( Y, P ){
  predictions = P;
  labels = Y;
  sortObj = sort.int(predictions, index.return=TRUE)
  predictions = sortObj$x
  labels = labels[sortObj$ix]
  ordered = cbind(predictions,labels)
  N = length(predictions);
  rest = N%%10;
  S=rep(0,10)
  W=rep(0,10)
  for (i in 1:10){
    if (i <= rest){
      startIdx = as.integer((i-1) * ceiling(N / 10) + 1)
      endIdx = as.integer(i * ceiling(N / 10))
    }else{
      startIdx = as.integer(rest + (i-1)*floor(N/10)+1)
      endIdx = as.integer(rest + i*floor(N/10))    
    }
    group = ordered[startIdx:endIdx,];
    
    n = dim(group)[1];
    observed = mean(group[,2]);
    expected = mean(group[,1]);
    S[i] = abs(expected-observed);
    W[i] = n/N;
  }
  res = sum(S*W);
  return(res)
}

getRMSE = function( Y, P ){
  res = sqrt(sum((Y-P)*(Y-P))/length(Y));
}

elbow = function(scores, alpha){
  b = length(scores);
  sigma2 = var(scores);
  sortObj = sort.int(scores, index.return=TRUE, decreasing = TRUE)
  R = sortObj$x
  idxs = sortObj$ix

  k = 1;
  while (R[k]==R[k+1]){
    k = k + 1;
  }
  while ((k<b)&&((R[k]-R[k+1])/sigma2 > alpha)){
    k = k+1;
  }
  
  if (k  > 1){
    res = idxs[1:k-1];
  }else{
    res = idxs[1];
  }
  return(res)
}

processModel = function (inModel, idxs){
  outModel = list()
  outModel$z = inModel$z;
  outModel$y = inModel$y;
  outModel$uy = inModel$uy;
  outModel$SV = inModel$SV[idxs];
  outModel$df = inModel$df[idxs];
  outModel$lambda = inModel$lambda[idxs];
  outModel$beta_org = inModel$beta_org[,idxs];
  outModel$beta = inModel$beta[,idxs];
  outModel$maxScoreIdx = 1;
  outModel$minScoreIdx = length(idxs); 
  return(outModel)
}

#'@title Computer evaluation measures for binary classification scores
#'@author Mahdi Pakdaman
#'@param PTE: vector predicted of classification scores, YTE: corresponding true class \{0,1\}
#'@export
enir.getMeasures = function(PTE, YTE ){
#   GETMEASURES Summary of this function goes here
  if (sum(!is.finite(PTE))>0){
    print('there are some nan value in predictions')
  }
  
  if (sum(!is.finite(YTE))>0){
    print('there are some nan value in predictions')
  }
  
  
  idx = is.finite(YTE)&is.finite(PTE);
  YTE = YTE[idx]; PTE = PTE[idx];
  
  res = list();
  res$RMSE = getRMSE(YTE,PTE);
  res$AUC = getAUC(YTE,PTE);
  res$ACC = sum(YTE==(PTE>=0.5))/length(YTE);    
  res$MCE = getMCE(YTE,PTE); #Computing the Max Calibratio Error among all bins 
  res$ECE = getECE(YTE,PTE); #Computing Average Calinration Error on different binns
  return(res)
}

#'@title Create ENIR model
#'@author Mahdi Pakdaman
#'@param y: uncalibrated classification scores y [0,1], z: corresponding true class z \{0,1\}
#'@export
enir.build = function(y, z, scoreFunc='BIC', alpha = 0.005){
  library(neariso)
  #sort instances based on the y score
  sortObj = sort.int(y, index.return=TRUE)
  y = sortObj$x
  z = z[sortObj$ix]
  #not complete yet
  uObj = uniqueInSortedArray(y)
  uy = uObj$ux
  Iy = uObj$Ix
  Iuy = uObj$Iux  

  stopifnot(sum(uy[Iuy]!= y)==0)
  stopifnot(sum(y[Iy]!= uy)==0)
  stopifnot(sum(uy!= unique(y))==0)
  
  res <- neariso2(z, y, lambda=NULL)
  lambda =  res$lambda
  beta = res$beta
  df = res$df

  beta = round(beta*10^6)/10^6
  #make sure there is nothing larger than 1
  idx1 = (beta >1)
  beta[idx1] = 1
  #make sure there is nothing less than 0
  idx0 = (beta<0)
  beta[idx0] = 0
  
  df = getDF(beta)
  score = (1:length(lambda))*0
  for(i in 1:length(lambda)) {
    p = beta[,i]
    score[i] = getScoreLocal(z, p[Iuy], df[i],scoreFunc)
  }
  
#   Remove those that have infinit score as well as the model associated
#   with lambda = 0 (dummy model)
  idx = is.finite(score) 
  if (length(idx)>1) {
    idx[1] = (1==0)
  }

  df = df[idx];
  lambda = lambda[idx];
  beta = beta[,idx];
  score = score[idx];
  
  MNM = length(df); # Maximum Number of Models
  maxScore = -Inf;
  maxScoreIdx = 0;
  minScore = Inf;
  minScoreIdx = 0;
  SV = (1:MNM)*0;
  for (b in 1:MNM){
    SV[b] = score[b]
    if (score[b]>maxScore){ 
      maxScoreIdx = b
      maxScore = score[b]
    }
    if (score[b] < minScore){
      minScoreIdx = b;
      minScore = score[b];
    }
  }
  SV = exp((min(SV)-SV)/2); #Compute Reletive Likelihood

  model = list(maxScoreIdx = maxScoreIdx, minScoreIdx = minScoreIdx, SV = SV,
               z = z, y = y, uy = uy, df = df, beta_org = beta, lambda = lambda, 
               Iy = Iy, Iuy = Iuy)
  # This part is added to smooth the bin estimates using beta prior similar to what we did in BBQ 
  model$beta = smoothBeta(model);

#   We can do shoulder method as we did for BBQ
  model2 = model;
  if (MNM > 1){
    idxs = elbow(SV, alpha);
    model2 = processModel(model, idxs);
  }    
  
  enir = list()
  enir$model = model;
  enir$prunedModel = model2; 
  return(enir)
}


#'@title predict calibrated estimate for an uncalibrated score using previously built ENIR model
#'@author Mahdi Pakdaman
#'@param y: uncalibrated classification scores y [0,1], z: corresponding true class z \{0,1\}
#'@export
enir.predict = function( enir, pin, option = 1 ){
#   This function used for calibrating the probabilitiesusing ENIR model
#   Input: 
#          - ENIR: the ENIR model learnt by lnEFBB
#          - pin : vector of Uncalibrated probabilities
#          - option: 0 use model selection, 1 use model Averaging   
#    Output:
#            - out : vector of calibrated probabilities

  
#        pout = zeros(length(pin),1);
#        enirModel = enir.prunedModel;
  enirModel = enir$model;
  pout = rep(0, length(pin))
    
  if (option == 1) {
#      Use model Averaging
    for (i in 1:length(pin)){
      pout[i] = getMA(enirModel,pin[i]);
    }
  }else if (option == 0){
#      Use Model Selection
    nirModel = list()
    nirModel$y = enirModel$y;
    nirModel$uy = enirModel$uy;
    nirModel$z = enirModel$z;
    nirModel$df = enirModel$df[1];
    nirModel$beta = enirModel$beta[,1];
    nirModel$lambda = enirModel$lambda[1];
    for (i in 1:length(pin)){
      pout[i] = getMS(nirModel, pin[i]);
    }
  }
  return(pout)
  #        It has been handled in build_enir
  #        pout = max(pout,0);
  #        pout = min(pout,1);
}


